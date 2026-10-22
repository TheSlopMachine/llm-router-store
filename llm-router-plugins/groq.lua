--- @plugin Groq
--- @author TheSlopMachine
--- @version 5.4.0
--- @plugin_api 1.5
--- @description Groq API: chat/reasoning, Whisper transcription/translation, Orpheus text-to-speech
--- @allow_host api.groq.com

local BASE_URL = "https://api.groq.com/openai/v1"

-- Free-plan values from https://console.groq.com/docs/rate-limits.
-- Limits can vary by account/plan; model discovery cannot retrieve account quotas.
local RATE_LIMITS = {
  ["groq/compound"] = { rpm = 30, rpd = 250, tpm = 70000 },
  ["groq/compound-mini"] = { rpm = 30, rpd = 250, tpm = 70000 },
  ["openai/gpt-oss-120b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["openai/gpt-oss-20b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["openai/gpt-oss-safeguard-20b"] = { rpm = 3, rpd = 1000, tpm = 2000 },
  ["qwen/qwen3.6-27b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["qwen/qwen3.8-27b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["meta-llama/llama-prompt-guard-2-22m"] = { rpm = 30, rpd = 14400, tpm = 15000 },
  ["meta-llama/llama-prompt-guard-2-86m"] = { rpm = 30, rpd = 14400, tpm = 15000 },
  ["allam-2-7b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["canopylabs/orpheus-v1-english"] = { rpm = 10, rpd = 100, tpm = 1200 },
  ["canopylabs/orpheus-arabic-saudi"] = { rpm = 10, rpd = 100, tpm = 1200 },
  -- Whisper quotas are measured in audio seconds; tpm stays 0.
  ["whisper-large-v3"] = { rpm = 20, rpd = 2000, tpm = 0 },
  ["whisper-large-v3-turbo"] = { rpm = 20, rpd = 2000, tpm = 0 },
  ["distil-whisper-large-v3-en"] = { rpm = 20, rpd = 1000, tpm = 0 },
}
local DEFAULT_LIMITS = { rpm = 30, rpd = 1000, tpm = 8000 }
local AUDIO_DEFAULT_LIMITS = { rpm = 20, rpd = 2000, tpm = 0 }

local function api_key_of(data)
  if type(data) ~= "table" then return "" end
  return data.api_key or ""
end

-- Dead-key bench: shared disable when automation is on (visible in the
-- dashboard, global to every path), unified cooldown park otherwise.
local function bench(ctx, cred_id, reason, wait_secs)
  local cfg = ctx.provider_config or {}
  if cfg.disable_failed_credentials == true then
    llm_router.credentials.disable(cred_id, reason)
  else
    llm_router.credentials.park(cred_id, wait_secs, reason)
  end
end

local function parse_duration_secs(raw)
  if type(raw) ~= "string" then return nil end
  local n = tonumber(raw)
  if n and n > 0 then return math.ceil(n) end
  local total = 0
  local h = tonumber(raw:match("([%d%.]+)h")) or 0
  local m = tonumber(raw:match("([%d%.]+)m")) or 0
  local s = tonumber(raw:match("([%d%.]+)s")) or 0
  local ms = tonumber(raw:match("([%d%.]+)ms")) or 0
  total = h * 3600 + m * 60 + s + ms / 1000
  if total > 0 then return math.ceil(total) end
  return nil
end

local function retry_after_secs(resp)
  local headers = resp and resp.headers or {}
  local wait = parse_duration_secs(headers["retry-after"] or headers["retry_after"])
  if wait then return wait end
  -- Groq reset headers use durations such as "2m59.56s", not Unix timestamps.
  wait = parse_duration_secs(headers["x-ratelimit-reset-requests"])
  if wait then return wait end
  wait = parse_duration_secs(headers["x-ratelimit-reset-tokens"])
  return wait or 60
end

local function upstream_message(resp, fallback)
  local body = resp and resp.body
  if type(body) ~= "string" or body == "" then return fallback end
  local ok, parsed = pcall(json.decode, body)
  if ok and type(parsed) == "table" then
    if type(parsed.error) == "table" and type(parsed.error.message) == "string" then
      return parsed.error.message
    end
    if type(parsed.message) == "string" then return parsed.message end
  end
  return fallback
end

local function is_daily_quota(resp)
  local body = string.lower(tostring(resp and resp.body or ""))
  return string.find(body, "per day", 1, true) ~= nil
    or string.find(body, "daily quota", 1, true) ~= nil
    or string.find(body, "daily limit", 1, true) ~= nil
end

local function rate_limit_error(resp)
  local daily = is_daily_quota(resp)
  if daily then
    return {
      message = upstream_message(resp, "groq daily quota exhausted"),
      code = "insufficient_quota",
      status = 429,
    }, "groq daily quota exhausted"
  end
  return {
    message = upstream_message(resp, "groq rate limited"),
    code = "rate_limit",
    status = 429,
  }, "groq rate limited"
end

local function has(list, value)
  for _, item in ipairs(list or {}) do
    if item == value then return true end
  end
  return false
end

local ORPHEUS_VOICES = {
  ["canopylabs/orpheus-v1-english"] = {
    autumn = true, diana = true, hannah = true,
    austin = true, daniel = true, troy = true,
  },
  ["canopylabs/orpheus-arabic-saudi"] = {
    abdullah = true, fahad = true, sultan = true,
    lulwa = true, noura = true, aisha = true,
  },
}

local function utf8_length(value)
  local _, count = tostring(value or ""):gsub("[^\128-\191]", "")
  return count
end

local function pick_proxies(ctx, limit)
  -- Unconfigured providers go direct: only an explicit pool selection
  -- (dashboard proxy switch) routes through pooled exits.
  local pool = nil
  if ctx.provider_config and ctx.provider_config.proxy and ctx.provider_config.proxy.pool ~= "" then
    pool = ctx.provider_config.proxy.pool
  end
  -- The pool acquires exits on demand: it searches for a live proxy when none
  -- is ready and waits for one. A direct pool, the wait deadline or a
  -- cancelled request degrade to one direct attempt, never to silence.
  local res = llm_router.proxies.require({ pool = pool, limit = limit or 3, fallback = "direct" })
  if not res or #res.proxies == 0 then
    return { {} }
  end
  return res.proxies
end

-- Build the upstream payload from the OpenAI-shaped request table:
-- pass everything through, fix the model id, pin the stream flag.
-- Router-only fields (model_name, cache_key) never leave the plugin.
local function build_payload(request, stream)
  local payload = {}
  for k, v in pairs(request) do
    if k ~= "model_name" and k ~= "cache_key" then payload[k] = v end
  end
  payload.model = request.model_name
  payload.stream = stream
  if not stream then payload.stream_options = nil end

  -- The router accepts OpenRouter's reasoning envelope, but Groq expects flat
  -- reasoning_effort/reasoning_format fields rather than `reasoning: {...}`.
  local reasoning = payload.reasoning
  payload.reasoning = nil
  if type(reasoning) == "table" then
    local model_name = request.model_name or ""
    local is_qwen3 = string.find(model_name, "qwen3", 1, true) ~= nil
    if reasoning.enabled == false then
      -- Qwen3 families support `none`; GPT-OSS only documents low/medium/high.
      payload.reasoning_effort = is_qwen3 and "none" or nil
    elseif type(reasoning.effort) == "string" and reasoning.effort ~= "" and reasoning.effort ~= "default" then
      payload.reasoning_effort = reasoning.effort
    elseif reasoning.enabled == true and is_qwen3
        and (payload.reasoning_effort == nil or payload.reasoning_effort == "default") then
      -- Groq's Qwen default is `none`, so an explicit enabled=true needs an effort.
      payload.reasoning_effort = "medium"
    end
    if reasoning.exclude == true and payload.reasoning_format == nil then
      payload.reasoning_format = "hidden"
    end
  end
  -- `default` means use provider defaults; omitting it is more portable than
  -- sending an effort literal unsupported by some model families.
  if payload.reasoning_effort == "default" then payload.reasoning_effort = nil end
  return payload
end

llm_router.register("groq", {
  icon = "https://console.groq.com/favicon.ico",

  proxy_schema = {},

  locales = {
    ["en"] = {
      ["groq.subtitle"] = "Get your API key from the Groq console.",
      ["groq.open_console"] = "Open Groq Console",
      ["misc.api_key"] = "API Key",
      ["misc.save"] = "Save",
    },
    ["ru"] = {
      ["groq.subtitle"] = "Получите API-ключ в консоли Groq.",
      ["groq.open_console"] = "Открыть консоль Groq",
      ["misc.api_key"] = "API-ключ",
      ["misc.save"] = "Сохранить",
    },
  },

  credential_schema = {
    {
      type = "section",
      title = "Groq",
      subtitle = "groq.subtitle",
      content = {
        { type = "link", text = "groq.open_console", url = "https://console.groq.com/keys" },
        { type = "secret", name = "api_key", label = "misc.api_key", required = true },
        { type = "button", text = "misc.save", form_action = "submit" },
      },
    },
  },

  validate_credentials = function(data)
    local key = data.api_key
    if key == nil or key == "" then
      return false, { message = "api_key is required", code = "invalid_request_error" }
    end
    if #key < 20 then return false, { message = "api_key: minimum 20 characters", code = "invalid_request_error" } end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, { message = "no groq credentials configured", code = "invalid_request_error", status = 400 }
    end
    local client = llm_router.http_client({})
    local resp, err = client:request({
      method = "GET",
      url = BASE_URL .. "/models",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(creds[1].data) },
    })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, {
        message = "groq model discovery failed with status " .. tostring(resp.status),
        code = "server_error",
        status = resp.status,
      }
    end
    local decoded, page = pcall(json.decode, resp.body or "")
    if not decoded or type(page) ~= "table" then
      return nil, { message = "groq model discovery returned invalid JSON", code = "server_error", status = 502 }
    end

    local infos = {}
    for _, m in ipairs(page.data or {}) do
      local input = m.input_modalities or {}
      local output = m.output_modalities or {}
      local name = (m.name and m.name ~= "") and m.name or m.id

      if m.active and has(input, "audio")
          and (has(output, "text") or has(output, "transcription")) then
        local limits = RATE_LIMITS[m.id] or AUDIO_DEFAULT_LIMITS
        local endpoints = { "audio/transcriptions" }
        -- Groq documents translation only for whisper-large-v3, not Turbo.
        if m.id == "whisper-large-v3" then
          table.insert(endpoints, "audio/translations")
        end
        table.insert(infos, {
          name = m.id,
          display_name = name,
          rpm = limits.rpm,
          tpm = limits.tpm,
          rpd = limits.rpd,
          input_modalities = input,
          output_modalities = output,
          endpoints = endpoints,
        })
      elseif m.active and has(input, "text")
          and (has(output, "speech") or has(output, "audio")) then
        local limits = RATE_LIMITS[m.id] or DEFAULT_LIMITS
        table.insert(infos, {
          name = m.id,
          display_name = name,
          rpm = limits.rpm,
          tpm = limits.tpm,
          rpd = limits.rpd,
          input_modalities = input,
          output_modalities = output,
          endpoints = { "audio/speech" },
        })
      elseif m.active and has(input, "text") and has(output, "text") then
        local limits = RATE_LIMITS[m.id] or DEFAULT_LIMITS
        local params = {}
        for _, p in ipairs(m.supported_sampling_parameters or {}) do
          table.insert(params, p)
        end
        if has(m.supported_features, "tools") then
          table.insert(params, "tools")
          table.insert(params, "tool_choice")
        end
        if has(m.supported_features, "json_mode") then table.insert(params, "response_format") end
        if has(m.supported_features, "structured_outputs") then table.insert(params, "structured_outputs") end

        local reasoning = nil
        if has(m.supported_features, "reasoning") then
          table.insert(params, "reasoning")
          table.insert(params, "reasoning_effort")
          if m.id == "openai/gpt-oss-20b" or m.id == "openai/gpt-oss-120b"
              or m.id == "openai/gpt-oss-safeguard-20b" then
            reasoning = {
              supported_efforts = { "high", "medium", "low" },
              default_effort = "medium",
              default_enabled = true,
            }
          elseif m.id == "qwen/qwen3.8-27b" then
            reasoning = {
              supported_efforts = { "high", "medium", "low", "default", "none" },
              default_effort = "none",
              default_enabled = false,
            }
          else
            reasoning = {
              supported_efforts = { "default", "none" },
              default_effort = "default",
              default_enabled = true,
            }
          end
        end
        table.insert(infos, {
          name = m.id,
          display_name = name,
          rpm = limits.rpm,
          tpm = limits.tpm,
          rpd = limits.rpd,
          context_window = m.context_window or 0,
          max_tokens = m.max_completion_tokens or 0,
          input_modalities = input,
          output_modalities = output,
          supported_parameters = params,
          reasoning = reasoning,
          endpoints = { "chat/completions" },
        })
      end
    end
    table.sort(infos, function(a, b) return a.name < b.name end)
    return infos
  end,

  -- Account liveness: the keyed /models listing answers 200 for live keys
  -- and 401 for dead ones. Only 401 verdicts unhealthy; every other
  -- failure is unknown and changes nothing.
  check_health = function(ctx, credential)
    local key = api_key_of(credential.data)
    if key == "" then return { status = "unhealthy", message = "api_key is required" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local resp, err = client:request({
      method = "GET",
      url = BASE_URL .. "/models",
      headers = { ["Authorization"] = "Bearer " .. key },
    })
    if err then
      local message = "request failed"
      if type(err) == "table" and type(err.message) == "string" then message = err.message end
      return { status = "unknown", message = message }
    end
    if resp.status == 200 then return { status = "healthy" } end
    if resp.status == 401 then return { status = "unhealthy", message = "api key rejected" } end
    return { status = "unknown", message = "status " .. tostring(resp.status) }
  end,

  -- Model liveness: per-model metadata GET proves existence with zero
  -- generation spend. Direct, no proxy: pool health must not decide model
  -- health. Retries across credentials with unanimity for death: healthy
  -- wins at once, quota outranks unknown, unhealthy needs every tried key
  -- to report model-missing. Probes never park or disable.
  check_model = function(ctx, model_ref)
    local endpoint = model_ref.endpoint or "chat"
    -- Vision-capable chat models still serve chat/completions upstream;
    -- vision is a dashboard-side input distinction, not a separate API.
    if endpoint ~= "chat" and endpoint ~= "vision" and endpoint ~= "transcription"
        and endpoint ~= "translation" and endpoint ~= "speech"
        and endpoint ~= "audio/transcriptions" and endpoint ~= "audio/translations"
        and endpoint ~= "audio/speech" then
      return { status = "unknown", message = "endpoint not supported: " .. tostring(endpoint) }
    end
    local model_name = model_ref.model_name or ""
    if model_name == "" then return { status = "unknown", message = "model name is required" } end
    local creds = llm_router.credentials.list()
    if #creds == 0 then return { status = "unknown", message = "no credentials available" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local missing = 0
    local tried = 0
    local seen_quota = false
    local seen_unknown = nil
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key == "" then
        seen_unknown = seen_unknown or "api_key is required"
      else
        tried = tried + 1
        llm_router.log("debug", "probing model", { cred_id = cred.id, model = model_ref.model, endpoint = endpoint })
        local resp, err = client:request({
          method = "GET",
          url = BASE_URL .. "/models/" .. model_name,
          headers = { ["Authorization"] = "Bearer " .. key },
        })
        if err then
          local message = "request failed"
          if type(err) == "table" and type(err.message) == "string" then message = err.message end
          seen_unknown = seen_unknown or message
        elseif resp.status == 200 then
          return { status = "healthy", response = "ok" }
        elseif resp.status == 404 then
          missing = missing + 1
          seen_unknown = seen_unknown or "model not found upstream"
        elseif resp.status == 429 then
          seen_quota = true
        elseif resp.status == 400 then
          local blocked = false
          local ok, page = pcall(json.decode, resp.body or "")
          if ok and type(page) == "table" and type(page.error) == "table" and page.error.code == "blocked_api_access" then
            blocked = true
          end
          if blocked then
            seen_quota = true
          else
            seen_unknown = seen_unknown or ("status " .. tostring(resp.status))
          end
        elseif resp.status == 401 or resp.status == 403 then
          seen_unknown = seen_unknown or "api key rejected"
        else
          seen_unknown = seen_unknown or ("status " .. tostring(resp.status))
        end
      end
    end
    if tried > 0 and missing == tried then
      return { status = "unhealthy", message = "model not found upstream" }
    end
    if seen_quota then
      return { status = "quota_exceeded", message = "quota exhausted" }
    end
    return { status = "unknown", message = seen_unknown or "health check inconclusive" }
  end,

  complete = function(ctx, request)
    local client = llm_router.http_client({})
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      llm_router.log("debug", "trying credential", { cred_id = cred.id, model = request.model })
      local key = api_key_of(cred.data)
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local resp, err = client:request({
          method = "POST",
          url = BASE_URL .. "/chat/completions",
          headers = { ["Authorization"] = "Bearer " .. key, ["Content-Type"] = "application/json" },
          body = json.encode(build_payload(request, false)),
          proxy_url = px.url,
        })
        if err == nil and resp.status == 200 then
          local ok, out = pcall(json.decode, resp.body or "")
          if not ok or type(out) ~= "table" then
            return nil, { message = "groq returned invalid JSON", code = "server_error", status = 502 }
          end
          out.model = request.model
          return out
        elseif err ~= nil then
          llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
          last_err = err
        elseif resp.status == 401 then
          bench(ctx, cred.id, "groq rejected the api key", 300)
          last_err = { message = upstream_message(resp, "groq rejected the api key"), code = "authentication_error", status = 401 }
          llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
          break
        elseif resp.status == 403 then
          llm_router.log("debug", "egress blocked", { status = 403, code = "server_error", cred_id = cred.id })
          last_err = { message = upstream_message(resp, "groq egress block on current exit"), code = "server_error", status = 502 }
        elseif resp.status == 429 then
          local quota_err, park_reason = rate_limit_error(resp)
          llm_router.credentials.park(cred.id, retry_after_secs(resp), park_reason)
          llm_router.log("warn", park_reason, { status = 429, code = quota_err.code, cred_id = cred.id })
          last_err = quota_err
          break
        elseif resp.status == 400 or resp.status == 404 or resp.status == 422 then
          local message = upstream_message(resp, "groq rejected the request with status " .. tostring(resp.status))
          llm_router.log("error", "request rejected", { status = resp.status, code = "invalid_request_error", cred_id = cred.id })
          return nil, { message = message, code = "invalid_request_error", status = resp.status }
        else
          llm_router.log("debug", "unexpected status", { status = resp.status, code = "server_error", cred_id = cred.id })
          last_err = {
            message = upstream_message(resp, "groq returned status " .. tostring(resp.status)),
            code = "server_error",
            status = resp.status,
          }
        end
      end
    end
    local exhausted = last_err or { message = "all groq credentials exhausted", code = "server_error" }
    llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
    return nil, exhausted
  end,

  complete_stream = function(ctx, request, emit)
    local client = llm_router.http_client({})
    local full_model = request.model
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      llm_router.log("debug", "trying credential", { cred_id = cred.id, model = request.model })
      local key = api_key_of(cred.data)
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local done = false
        local next_cred = false
        local terminal_err = nil
        local resp, stream_err = client:stream({
          method = "POST",
          url = BASE_URL .. "/chat/completions",
          headers = { ["Authorization"] = "Bearer " .. key, ["Content-Type"] = "application/json" },
          body = json.encode(build_payload(request, true)),
          proxy_url = px.url,
          on_response = function(r)
            if r.status == 401 then
              bench(ctx, cred.id, "groq rejected the api key", 300)
              next_cred = true
              return { message = upstream_message(r, "groq rejected the api key"), code = "authentication_error", status = 401 }
            end
            if r.status == 403 then
              return { message = upstream_message(r, "groq egress block on current exit"), code = "server_error", status = 502 }
            end
            if r.status == 429 then
              local quota_err, park_reason = rate_limit_error(r)
              llm_router.credentials.park(cred.id, retry_after_secs(r), park_reason)
              next_cred = true
              llm_router.log("warn", park_reason, { status = 429, code = quota_err.code, cred_id = cred.id })
              return quota_err
            end
            if r.status == 400 or r.status == 404 or r.status == 422 then
              terminal_err = {
                message = upstream_message(r, "groq rejected the request with status " .. tostring(r.status)),
                code = "invalid_request_error",
                status = r.status,
              }
              return terminal_err
            end
            if r.status ~= 200 then
              return {
                message = upstream_message(r, "groq returned status " .. tostring(r.status)),
                code = "server_error",
                status = r.status,
              }
            end
          end,
          on_line = function(line)
            if line:sub(1, 6) ~= "data: " then return end
            local data = line:sub(7)
            if data == "[DONE]" then return end
            local ok, chunk = pcall(json.decode, data)
            if not ok or type(chunk) ~= "table" then return end
            local has_choices = type(chunk.choices) == "table" and #chunk.choices > 0
            if not has_choices and chunk.usage == nil then return end
            chunk.model = full_model
            emit(chunk)
            done = true
          end,
        })
        if terminal_err then return nil, terminal_err end
        if stream_err then
          last_err = stream_err
        else
          return
        end
        if next_cred then break end
        -- A stream that emitted already belongs to that attempt: surface
        -- instead of failing over mid-stream.
        if done then return nil, last_err end
      end
    end
    local exhausted = last_err or { message = "all groq credentials exhausted", code = "server_error" }
    llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
    return nil, exhausted
  end,

  -- Speech-to-text: always fetch verbose_json upstream; the router renders
  -- the client's response_format (json/text/srt/vtt) from the normalized
  -- response, so segments must be present when the client asked for them.
  transcribe = function(ctx, request)
    local task = request.task or "transcribe"
    if task ~= "transcribe" and task ~= "translate" then
      return nil, { message = "unsupported audio task: " .. tostring(task), code = "invalid_request_error", status = 400 }
    end
    if task == "translate" and request.language and request.language ~= "" and request.language ~= "en" then
      return nil, { message = "Groq audio translations only accept language=en", code = "invalid_request_error", status = 400 }
    end

    local client = llm_router.http_client({ timeout_ms = 300000 })
    local parts = {
      { name = "model", value = request.model_name },
      { name = "file", filename = request.file_name, content_type = request.content_type, data = request.file },
      { name = "response_format", value = "verbose_json" },
    }
    if request.language and request.language ~= "" then parts[#parts + 1] = { name = "language", value = request.language } end
    if request.prompt and request.prompt ~= "" then parts[#parts + 1] = { name = "prompt", value = request.prompt } end
    if request.temperature ~= nil then parts[#parts + 1] = { name = "temperature", value = tostring(request.temperature) } end
    local gran = {}
    local function add_gran(g)
      for _, x in ipairs(gran) do
        if x == g then return end
      end
      gran[#gran + 1] = g
    end
    if request.needs_segments then add_gran("segment") end
    for _, g in ipairs(request.timestamp_granularities or {}) do add_gran(g) end
    for _, g in ipairs(gran) do
      parts[#parts + 1] = { name = "timestamp_granularities[]", value = g }
    end
    local body, ctype = llm_router.multipart(parts)
    local path = task == "translate" and "/audio/translations" or "/audio/transcriptions"
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      llm_router.log("debug", "trying credential", { cred_id = cred.id, model = request.model, task = task })
      local key = api_key_of(cred.data)
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local resp, err = client:request({
          method = "POST",
          url = BASE_URL .. path,
          headers = { ["Authorization"] = "Bearer " .. key, ["Content-Type"] = ctype },
          body = body,
          proxy_url = px.url,
        })
        if err == nil and resp.status == 200 then
          local ok, out = pcall(json.decode, resp.body or "")
          if not ok or type(out) ~= "table" then
            return nil, { message = "groq returned invalid transcription JSON", code = "server_error", status = 502 }
          end
          return out
        elseif err ~= nil then
          llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
          last_err = err
        elseif resp.status == 401 then
          bench(ctx, cred.id, "groq rejected the api key", 300)
          last_err = { message = upstream_message(resp, "groq rejected the api key"), code = "authentication_error", status = 401 }
          llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
          break
        elseif resp.status == 429 then
          local quota_err, park_reason = rate_limit_error(resp)
          llm_router.credentials.park(cred.id, retry_after_secs(resp), park_reason)
          last_err = quota_err
          llm_router.log("warn", park_reason, { status = 429, code = quota_err.code, cred_id = cred.id })
          break
        elseif resp.status == 400 or resp.status == 404 or resp.status == 422 then
          local message = upstream_message(resp, "groq rejected the request with status " .. tostring(resp.status))
          llm_router.log("error", "request rejected", { status = resp.status, code = "invalid_request_error", cred_id = cred.id })
          return nil, { message = message, code = "invalid_request_error", status = resp.status }
        else
          llm_router.log("debug", "unexpected status", { status = resp.status, code = "server_error", cred_id = cred.id })
          last_err = {
            message = upstream_message(resp, "groq returned status " .. tostring(resp.status)),
            code = "server_error",
            status = resp.status,
          }
        end
      end
    end
    local exhausted = last_err or { message = "all groq credentials exhausted", code = "server_error" }
    llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
    return nil, exhausted
  end,

  -- Orpheus TTS only supports WAV, requires a model-specific voice, and limits
  -- input to 200 characters. English vocal directions are expressed inline.
  speech = function(ctx, request)
    local model = request.model_name or ""
    local voices = ORPHEUS_VOICES[model]
    if not voices then
      return nil, { message = "unsupported Groq speech model: " .. model, code = "invalid_request_error", status = 400 }
    end
    local text = tostring(request.input or "")
    local voice = tostring(request.voice or "")
    if text == "" then
      return nil, { message = "input is required", code = "invalid_request_error", status = 400 }
    end
    if voice == "" then
      return nil, { message = "voice is required for Groq Orpheus", code = "invalid_request_error", status = 400 }
    end
    if not voices[voice] then
      return nil, { message = "voice is not supported by " .. model, code = "invalid_request_error", status = 400 }
    end
    local format = request.response_format
    if format ~= nil and format ~= "" and format ~= "wav" then
      return nil, { message = "Groq Orpheus supports only response_format=wav", code = "invalid_request_error", status = 400 }
    end
    if request.speed ~= nil and tonumber(request.speed) and tonumber(request.speed) ~= 1 then
      return nil, { message = "Groq Orpheus does not support the speed parameter", code = "invalid_request_error", status = 400 }
    end
    if request.instructions and request.instructions ~= "" then
      if model ~= "canopylabs/orpheus-v1-english" then
        return nil, { message = "vocal directions are supported only by the English Orpheus model", code = "invalid_request_error", status = 400 }
      end
      local direction = tostring(request.instructions):gsub("^%[", ""):gsub("%]$", "")
      text = "[" .. direction .. "] " .. text
    end
    if utf8_length(text) > 200 then
      return nil, { message = "Groq Orpheus input is limited to 200 characters", code = "invalid_request_error", status = 400 }
    end

    local body = json.encode({ model = model, input = text, voice = voice, response_format = "wav" })
    local client = llm_router.http_client({ timeout_ms = 300000 })
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      llm_router.log("debug", "trying credential", { cred_id = cred.id, model = request.model })
      local key = api_key_of(cred.data)
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local resp, err = client:request({
          method = "POST",
          url = BASE_URL .. "/audio/speech",
          headers = { ["Authorization"] = "Bearer " .. key, ["Content-Type"] = "application/json" },
          body = body,
          proxy_url = px.url,
        })
        if err == nil and resp.status == 200 then
          if type(resp.body) ~= "string" or #resp.body == 0 then
            return nil, { message = "groq returned empty audio", code = "server_error", status = 502 }
          end
          return { audio_b64 = llm_router.base64_encode(resp.body), format = "wav" }
        elseif err ~= nil then
          last_err = err
          llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
        elseif resp.status == 401 then
          bench(ctx, cred.id, "groq rejected the api key", 300)
          last_err = { message = upstream_message(resp, "groq rejected the api key"), code = "authentication_error", status = 401 }
          llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
          break
        elseif resp.status == 403 then
          last_err = { message = upstream_message(resp, "groq egress block on current exit"), code = "server_error", status = 502 }
          llm_router.log("debug", "egress blocked", { status = 403, code = "server_error", cred_id = cred.id })
        elseif resp.status == 429 then
          local quota_err, park_reason = rate_limit_error(resp)
          llm_router.credentials.park(cred.id, retry_after_secs(resp), park_reason)
          last_err = quota_err
          llm_router.log("warn", park_reason, { status = 429, code = quota_err.code, cred_id = cred.id })
          break
        elseif resp.status == 400 or resp.status == 404 or resp.status == 422 then
          return nil, {
            message = upstream_message(resp, "groq rejected the speech request with status " .. tostring(resp.status)),
            code = "invalid_request_error",
            status = resp.status,
          }
        else
          last_err = {
            message = upstream_message(resp, "groq returned status " .. tostring(resp.status)),
            code = "server_error",
            status = resp.status,
          }
        end
      end
    end
    local exhausted = last_err or { message = "all groq credentials exhausted", code = "server_error" }
    llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
    return nil, exhausted
  end,
})
