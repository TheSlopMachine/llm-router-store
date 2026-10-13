--- @plugin Groq
--- @author TheSlopMachine
--- @version 5.2.1
--- @plugin_api 1.5
--- @description Groq OpenAI-compatible API: fast Llama/Qwen/gpt-oss inference, Whisper speech-to-text
--- @allow_host api.groq.com

local BASE_URL = "https://api.groq.com/openai/v1"

-- Free-plan rate limits from https://console.groq.com/docs/rate-limits
-- (Groq does not expose limits via the API; unknown models get conservative defaults).
local RATE_LIMITS = {
  ["groq/compound"] = { rpm = 30, rpd = 250, tpm = 70000 },
  ["groq/compound-mini"] = { rpm = 30, rpd = 250, tpm = 70000 },
  ["openai/gpt-oss-120b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["openai/gpt-oss-20b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["openai/gpt-oss-safeguard-20b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["qwen/qwen3.6-27b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["qwen/qwen3.8-27b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  ["meta-llama/llama-prompt-guard-2-22m"] = { rpm = 30, rpd = 14400, tpm = 15000 },
  ["meta-llama/llama-prompt-guard-2-86m"] = { rpm = 30, rpd = 14400, tpm = 15000 },
  ["allam-2-7b"] = { rpm = 30, rpd = 1000, tpm = 8000 },
  -- Whisper STT: audio-seconds quotas instead of tokens; tpm stays 0.
  ["whisper-large-v3"] = { rpm = 20, rpd = 2000, tpm = 0 },
  ["whisper-large-v3-turbo"] = { rpm = 20, rpd = 2000, tpm = 0 },
  ["distil-whisper-large-v3-en"] = { rpm = 20, rpd = 1000, tpm = 0 },
}
local DEFAULT_LIMITS = { rpm = 30, rpd = 1000, tpm = 8000 }

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

local function retry_after_secs(resp)
  local headers = resp.headers or {}
  local raw = headers["retry-after"] or headers["retry_after"]
  local n = tonumber(raw)
  if n ~= nil and n > 0 then return math.floor(n) end
  return 60
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
      return nil,
        {
          message = "groq model discovery failed with status " .. tostring(resp.status),
          code = "server_error",
          status = resp.status,
        }
    end
    local page = json.decode(resp.body)
    local infos = {}
    for _, m in ipairs(page.data or {}) do
      -- Split by modality: audio-in/text-out models are speech-to-text,
      -- text-in/text-out are chat. Text-in/audio-out (playai-tts) waits for
      -- the speech endpoint.
      local function has(list, v)
        for _, x in ipairs(list or {}) do
          if x == v then return true end
        end
        return false
      end
      if m.active and has(m.input_modalities, "audio") and has(m.output_modalities, "text") then
        local limits = RATE_LIMITS[m.id] or { rpm = 20, rpd = 2000, tpm = 0 }
        table.insert(infos, {
          name = m.id,
          display_name = (m.name and m.name ~= "") and m.name or m.id,
          rpm = limits.rpm,
          tpm = limits.tpm,
          rpd = limits.rpd,
          input_modalities = m.input_modalities,
          output_modalities = m.output_modalities,
          endpoints = { "audio/transcriptions" },
        })
      elseif m.active and has(m.input_modalities, "text") and has(m.output_modalities, "text") then
        local limits = RATE_LIMITS[m.id] or DEFAULT_LIMITS
        -- OpenRouter-style parameter list: sampling params + feature flags.
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
        -- Measured against the live API: gpt-oss accepts graded effort,
        -- qwen3 is on/off only. Efforts are listed highest first.
        local reasoning = nil
        if has(m.supported_features, "reasoning") then
          table.insert(params, "reasoning")
          table.insert(params, "reasoning_effort")
          if m.id:find("gpt%-oss") then
            reasoning = {
              supported_efforts = { "max", "xhigh", "high", "medium", "low", "minimal", "default", "none" },
              default_effort = "default",
              default_enabled = true,
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
          display_name = (m.name and m.name ~= "") and m.name or m.id,
          rpm = limits.rpm,
          tpm = limits.tpm,
          rpd = limits.rpd,
          context_window = m.context_window or 0,
          max_tokens = m.max_completion_tokens or 0,
          input_modalities = m.input_modalities,
          output_modalities = m.output_modalities,
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
    if endpoint ~= "chat" and endpoint ~= "vision" and endpoint ~= "transcription" then
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
          local out = json.decode(resp.body)
          out.model = request.model
          return out
        elseif err ~= nil then
          llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
          last_err = err
        elseif resp.status == 401 then
          -- Groq answers 401 for bad keys: bench the credential.
          bench(ctx, cred.id, "groq rejected the api key", 300)
          llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
          break
        elseif resp.status == 403 then
          -- 403 ("Forbidden", "Access denied. Please check your network
          -- settings.") is the Cloudflare egress-IP block: the exit is
          -- at fault, not the credential. Try the next proxy.
          llm_router.log("debug", "egress blocked", { status = 403, code = "server_error", cred_id = cred.id })
          last_err = { message = "groq egress block on current exit", code = "server_error", status = 502 }
        elseif resp.status == 429 then
          local lower = string.lower(tostring(resp.body or ""))
          if string.find(lower, "per day", 1, true) then
            llm_router.credentials.park(cred.id, retry_after_secs(resp), "groq daily quota exhausted")
            llm_router.log("warn", "quota exhausted", { status = 429, code = "insufficient_quota", cred_id = cred.id })
            break
          end
          llm_router.credentials.park(cred.id, retry_after_secs(resp), "groq rate limited")
          llm_router.log("warn", "rate limited", { status = 429, code = "rate_limit", cred_id = cred.id })
          break
        elseif resp.status == 400 or resp.status == 404 or resp.status == 422 then
          llm_router.log("error", "request rejected", { status = resp.status, code = "invalid_request_error", cred_id = cred.id })
          return nil,
            {
              message = "groq rejected the request with status " .. tostring(resp.status),
              code = "invalid_request_error",
              status = resp.status,
            }
        else
          llm_router.log("debug", "unexpected status", { status = resp.status, code = "server_error", cred_id = cred.id })
          last_err = {
            message = "groq returned status " .. tostring(resp.status),
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
              llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
              return { message = "groq rejected the api key", code = "authentication_error", status = 401 }
            end
            if r.status == 403 then
              llm_router.log("debug", "egress blocked", { status = 403, code = "server_error", cred_id = cred.id })
              return { message = "groq egress block on current exit", code = "server_error", status = 502 }
            end
            if r.status == 429 then
              llm_router.credentials.park(cred.id, retry_after_secs(r), "groq rate limited")
              next_cred = true
              llm_router.log("warn", "rate limited", { status = 429, code = "rate_limit", cred_id = cred.id })
              return { message = "groq rate limited", code = "rate_limit", status = 429 }
            end
            if r.status ~= 200 then
              llm_router.log("error", "unexpected status", { status = r.status, code = "server_error", cred_id = cred.id })
              return {
                message = "groq returned status " .. tostring(r.status),
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
            -- Forward the usage-only final chunk (include_usage): the emit
            -- contract accepts chunks without choices when usage is present.
            local has_choices = type(chunk.choices) == "table" and #chunk.choices > 0
            if not has_choices and chunk.usage == nil then return end
            chunk.model = full_model
            emit(chunk)
            done = true
          end,
        })
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
    local client = llm_router.http_client({ timeout_ms = 300000 })
    local parts = {
      { name = "model", value = request.model_name },
      { name = "file", filename = request.file_name, content_type = request.content_type, data = request.file },
      { name = "response_format", value = "verbose_json" },
    }
    if request.language then parts[#parts + 1] = { name = "language", value = request.language } end
    if request.prompt then parts[#parts + 1] = { name = "prompt", value = request.prompt } end
    if request.temperature then parts[#parts + 1] = { name = "temperature", value = tostring(request.temperature) } end
    local gran = {}
    local function add_gran(g)
      for _, x in ipairs(gran) do
        if x == g then return end
      end
      gran[#gran + 1] = g
    end
    if request.needs_segments then add_gran("segment") end
    for _, g in ipairs(request.timestamp_granularities or {}) do
      add_gran(g)
    end
    for _, g in ipairs(gran) do
      parts[#parts + 1] = { name = "timestamp_granularities[]", value = g }
    end
    local body, ctype = llm_router.multipart(parts)
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      llm_router.log("debug", "trying credential", { cred_id = cred.id, model = request.model })
      local key = api_key_of(cred.data)
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local resp, err = client:request({
          method = "POST",
          url = BASE_URL .. "/audio/transcriptions",
          headers = { ["Authorization"] = "Bearer " .. key, ["Content-Type"] = ctype },
          body = body,
          proxy_url = px.url,
        })
        if err == nil and resp.status == 200 then
          return json.decode(resp.body)
        elseif err ~= nil then
          llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
          last_err = err
        elseif resp.status == 401 then
          bench(ctx, cred.id, "groq rejected the api key", 300)
          llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
          break
        elseif resp.status == 429 then
          llm_router.credentials.park(cred.id, retry_after_secs(resp), "groq rate limited")
          llm_router.log("warn", "rate limited", { status = 429, code = "rate_limit", cred_id = cred.id })
          break
        elseif resp.status == 400 or resp.status == 404 or resp.status == 422 then
          llm_router.log("error", "request rejected", { status = resp.status, code = "invalid_request_error", cred_id = cred.id })
          return nil,
            {
              message = "groq rejected the request with status " .. tostring(resp.status),
              code = "invalid_request_error",
              status = resp.status,
            }
        else
          llm_router.log("debug", "unexpected status", { status = resp.status, code = "server_error", cred_id = cred.id })
          last_err = {
            message = "groq returned status " .. tostring(resp.status),
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
