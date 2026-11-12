--- @plugin Groq
--- @author TheSlopMachine
--- @version 5.5.1
--- @plugin_api 1.5
--- @description Groq API: chat/reasoning, Whisper transcription/translation, Orpheus text-to-speech
--- @allow_host api.groq.com

local BASE_URL = "https://api.groq.com/openai/v1"

-- Rate limits bind to the organization and the model. No table is kept:
-- every response carries x-ratelimit-remaining-* for the key+model pair, so
-- an exhausted daily request quota cools the pair before Groq answers 429,
-- and every other limit (RPM, TPM, TPD) cools it on the 429 itself.
local COOLDOWN_SCOPE = "rate_limit"
local DEFAULT_COOLDOWN_SECS = 60
local MAX_COOLDOWN_SECS = 86400
local TERMS_REASON = "terms"
local TERMS_COOLDOWN_SECS = 3600

local function api_key_of(data)
  if type(data) ~= "table" or type(data.api_key) ~= "string" then return "" end
  return (data.api_key:match("^%s*(.-)%s*$"))
end

-- How errors name a key: its dashboard label, else its id.
local function key_name(cred)
  local label = type(cred) == "table" and cred.label or nil
  if type(label) == "string" and label ~= "" then return "\"" .. label .. "\"" end
  return tostring(type(cred) == "table" and cred.id or "?")
end

local function error_of(message, code, status, headers)
  local e = { message = message, code = code or "server_error", status = status or 502 }
  if headers ~= nil then e.headers = headers end
  return e
end

local function invalid_request(message)
  return error_of(message, "invalid_request_error", 400)
end

local function retry_headers(secs)
  if type(secs) ~= "number" or secs < 1 then return nil end
  return { ["Retry-After"] = tostring(math.ceil(secs)) }
end

-- Dead-key bench: shared disable when automation is on (visible in the
-- dashboard, global to every path), unified cooldown park otherwise.
local function bench(ctx, cred_id, reason, wait_secs)
  local cfg = ctx.provider_config
  if type(cfg) == "table" and cfg.disable_failed_credentials == true then
    if pcall(llm_router.credentials.disable, cred_id, reason) then return end
  end
  pcall(llm_router.credentials.park, cred_id, wait_secs, reason)
end

-- Groq durations: "2m59.56s", "7.66s", "250ms", "1h2m", or plain seconds.
local function parse_duration_secs(raw)
  if type(raw) ~= "string" then return nil end
  local n = tonumber(raw)
  if n then return n > 0 and math.ceil(n) or nil end
  local scale = { h = 3600, m = 60, s = 1, ms = 0.001 }
  local total = 0
  for num, unit in raw:gmatch("([%d%.]+)(%a+)") do
    local value = tonumber(num)
    if value == nil or scale[unit] == nil then return nil end
    total = total + value * scale[unit]
  end
  if total > 0 then return math.ceil(total) end
  return nil
end

local function upstream_error(resp)
  local body = resp and resp.body
  if type(body) ~= "string" or body == "" then return nil end
  local ok, parsed = pcall(json.decode, body)
  if ok and type(parsed) == "table" and type(parsed.error) == "table" then return parsed.error end
  return nil
end

local function upstream_message(resp, fallback)
  local e = upstream_error(resp)
  if e and type(e.message) == "string" and e.message ~= "" then return e.message end
  return fallback
end

local function upstream_code(resp)
  local e = upstream_error(resp)
  return e and type(e.code) == "string" and e.code or nil
end

-- ---------------------------------------------------------------------------
-- Per key+model cooldowns
-- ---------------------------------------------------------------------------

-- Groq limits bind to the organization and the model, so a 429 cools only the
-- key+model pair (parking would hide the key from every model). Rows live in
-- plugin storage with a TTL.
local function pair_key(cred_id, model_name)
  return tostring(cred_id) .. ":" .. tostring(model_name)
end

local function set_cooldown(cred_id, model_name, secs, reason)
  secs = math.ceil(tonumber(secs) or DEFAULT_COOLDOWN_SECS)
  if secs < 1 then secs = 1 end
  if secs > MAX_COOLDOWN_SECS then secs = MAX_COOLDOWN_SECS end
  local now = os.time()
  pcall(llm_router.storage.set, COOLDOWN_SCOPE, pair_key(cred_id, model_name),
    { ["until"] = now + secs, reason = reason or "rate limited" }, { ttl = secs })
end

local function clear_cooldown(cred_id, model_name)
  pcall(llm_router.storage.delete, COOLDOWN_SCOPE, pair_key(cred_id, model_name))
end

local function cooldown_remaining(cred_id, model_name)
  local ok, value = pcall(llm_router.storage.get, COOLDOWN_SCOPE, pair_key(cred_id, model_name))
  if not ok or type(value) ~= "table" then return nil end
  local remaining = (tonumber(value["until"]) or 0) - os.time()
  if remaining > 0 then return remaining, value.reason end
  return nil
end

-- (usable, total, min_wait, all_terms): all_terms is true when every key
-- cools because its organization has not accepted the model's terms.
local function plan_credentials(model_name)
  local usable, total, min_wait, terms = {}, 0, nil, 0
  for _, cred in ipairs(llm_router.credentials.list()) do
    total = total + 1
    local remaining, reason = cooldown_remaining(cred.id, model_name)
    if remaining ~= nil then
      min_wait = min_wait and math.min(min_wait, remaining) or remaining
      if reason == TERMS_REASON then terms = terms + 1 end
    else
      usable[#usable + 1] = cred
    end
  end
  return usable, total, min_wait, (terms > 0 and terms == total)
end

local function terms_url(model_name)
  return "https://console.groq.com/playground?model=" .. tostring(model_name):gsub("/", "%%2F")
end

local function terms_error(model_name)
  return error_of("groq model " .. tostring(model_name)
    .. " requires accepting its terms; no configured key's organization has accepted them. The org admin accepts them at "
    .. terms_url(model_name), "model_unavailable", 403)
end

-- ---------------------------------------------------------------------------
-- Proxies
-- ---------------------------------------------------------------------------

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

-- One request through the selected pool, for every call that is not
-- generation. A 403 (egress block), a 5xx or a transport fault on one exit
-- moves on to the next; a direct pool yields one direct attempt.
local function pooled_request(ctx, client, request)
  local resp, err
  for _, px in ipairs(pick_proxies(ctx, 3)) do
    request.proxy_url = px.url
    resp, err = client:request(request)
    if err == nil and resp.status ~= 403 and resp.status < 500 then break end
  end
  return resp, err
end

-- ---------------------------------------------------------------------------
-- Upstream status classification
-- ---------------------------------------------------------------------------

local function is_daily_quota(resp)
  local body = string.lower(tostring(resp and resp.body or ""))
  return body:find("per day", 1, true) ~= nil or body:find("daily", 1, true) ~= nil
end

-- 429 wait: retry-after, then the body's "Please try again in 7.5s", then the
-- reset header of the exhausted dimension, then a minute.
local function rate_limit_wait(resp, daily)
  local headers = resp.headers or {}
  local wait = parse_duration_secs(headers["retry-after"])
  if wait then return wait end
  wait = parse_duration_secs(tostring(resp.body or ""):match("try again in ([%d%.hms]+)"))
  if wait then return wait end
  wait = parse_duration_secs(headers[daily and "x-ratelimit-reset-requests" or "x-ratelimit-reset-tokens"])
  return wait or DEFAULT_COOLDOWN_SECS
end

-- A served response that spent the last request of the day (remaining
-- requests 0; x-ratelimit-*-requests always count per day) cools the
-- key+model pair until the daily window refills, so the next call skips it.
local function note_limits(cred, model_name, resp)
  local headers = resp.headers or {}
  if headers["x-ratelimit-remaining-requests"] ~= "0" then return end
  local wait = parse_duration_secs(headers["x-ratelimit-reset-requests"]) or DEFAULT_COOLDOWN_SECS
  set_cooldown(cred.id, model_name, wait, "daily quota")
  llm_router.log("info", "groq daily requests spent", { cred_id = cred.id, model = model_name, wait = wait })
end

-- Actions: "ok" (value is the response), "proxy" (next exit, then next key),
-- "cred" (next key), "server" (next exit, never another key: an upstream
-- fault every key shares), "stop" (value is the final error).
local function classify(ctx, cred, model_name, resp, err)
  if err ~= nil then
    llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
    return "proxy", err
  end
  local status = resp.status
  if status == 200 then
    note_limits(cred, model_name, resp)
    return "ok", resp
  end
  local message = upstream_message(resp, "groq returned status " .. tostring(status))
  local code = upstream_code(resp)
  if status == 401 then
    bench(ctx, cred.id, "groq rejected the api key", 300)
    llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
    return "cred", error_of(message, "authentication_error", 401)
  end
  if status == 403 then
    llm_router.log("debug", "egress blocked", { status = 403, code = "server_error", cred_id = cred.id })
    return "proxy", error_of(message, "server_error", 502)
  end
  if status == 429 then
    local daily = is_daily_quota(resp)
    local wait = rate_limit_wait(resp, daily)
    set_cooldown(cred.id, model_name, wait, daily and "daily quota" or "rate limited")
    local e = error_of(message, daily and "insufficient_quota" or "rate_limit", 429, retry_headers(wait))
    llm_router.log("warn", daily and "groq daily quota exhausted" or "groq rate limited",
      { status = 429, code = e.code, cred_id = cred.id, model = model_name, wait = wait })
    return "cred", e
  end
  if code == "model_terms_required" then
    set_cooldown(cred.id, model_name, TERMS_COOLDOWN_SECS, TERMS_REASON)
    llm_router.log("warn", "model terms not accepted", { status = status, cred_id = cred.id, model = model_name })
    return "cred", terms_error(model_name)
  end
  if status == 404 then
    return "stop", error_of(message, "not_found", 404)
  end
  if status == 400 or status == 413 or status == 422 then
    llm_router.log("error", "request rejected", { status = status, code = "invalid_request_error", cred_id = cred.id })
    return "stop", error_of(message, "invalid_request_error", status)
  end
  if status == 498 or status == 503 then
    -- Capacity (flex 498, overload 503) is the backend's, not the key's.
    local wait = parse_duration_secs((resp.headers or {})["retry-after"]) or 5
    return "stop", error_of(message, "server_error", 503, retry_headers(wait))
  end
  llm_router.log("debug", "unexpected status", { status = status, code = "server_error", cred_id = cred.id })
  if status >= 500 then return "server", error_of(message, "server_error", status) end
  return "stop", error_of(message, "server_error", status)
end

-- Credential x exit loop shared by every generation path. send(cred, key,
-- proxy_url) returns an action and a value as classify does.
local function attempt(ctx, model_name, send)
  local usable, total, min_wait, all_terms = plan_credentials(model_name)
  if total == 0 then
    return nil, error_of("no groq credentials configured", "server_error", 503)
  end
  if #usable == 0 then
    if all_terms then return nil, terms_error(model_name) end
    return nil, error_of("every groq key is rate limited on " .. tostring(model_name), "rate_limit", 429, retry_headers(min_wait))
  end
  local last_err = nil
  for _, cred in ipairs(usable) do
    llm_router.log("debug", "trying credential", { cred_id = cred.id, model = model_name })
    local key = api_key_of(cred.data)
    local action, value = nil, nil
    for _, px in ipairs(pick_proxies(ctx, 3)) do
      action, value = send(cred, key, px.url)
      if action == "ok" then return value end
      if action == "stop" then return nil, value end
      last_err = value
      if action == "cred" then break end
    end
    if action == "server" then break end
  end
  local left, _, wait, terms = plan_credentials(model_name)
  if #left == 0 and terms then return nil, terms_error(model_name) end
  if #left == 0 and last_err and (last_err.code == "rate_limit" or last_err.code == "insufficient_quota") then
    last_err.headers = retry_headers(wait) or last_err.headers
  end
  local exhausted = last_err or error_of("all groq credentials exhausted", "server_error", 502)
  llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
  return nil, exhausted
end

-- ---------------------------------------------------------------------------
-- Reasoning
-- ---------------------------------------------------------------------------

-- Router scale none/low/medium/high/max onto Groq reasoning_effort (checked
-- live 2026-11-12). "default" means the model default: accepted, never
-- advertised, sent as nothing.
--   gpt-oss (all):  high, medium, low   (reasoning always runs)
--   qwen3.x:        high, medium, low, none (Groq default: none)
local EFFORT_ORDER = { "max", "high", "medium", "low", "none" }

local function effort_levels(model_name)
  local lower = tostring(model_name):lower()
  if lower:find("gpt%-oss") then
    return { high = "high", medium = "medium", low = "low" }, { default_effort = "medium", default_enabled = true, mandatory = true }
  end
  if lower:find("qwen3", 1, true) then
    return { high = "high", medium = "medium", low = "low", none = "none" }, { default_effort = "none", default_enabled = false }
  end
  return nil
end

local function effort_list(levels)
  local list = {}
  for _, effort in ipairs(EFFORT_ORDER) do
    if levels[effort] then list[#list + 1] = effort end
  end
  return list
end

local function reasoning_info(model_name)
  local levels, defaults = effort_levels(model_name)
  if not levels then return { default_enabled = true } end
  return {
    supported_efforts = effort_list(levels),
    default_effort = defaults.default_effort,
    default_enabled = defaults.default_enabled,
    mandatory = defaults.mandatory,
  }
end

-- (reasoning_effort to send or nil, error). reasoning_effort or
-- reasoning.effort; reasoning.enabled=false reads as "none" and
-- enabled=true on an off-by-default model as "medium".
local function reasoning_effort_for(request)
  local model_name = request.model_name or ""
  local effort = request.reasoning_effort
  local r = type(request.reasoning) == "table" and request.reasoning or {}
  if effort == nil and type(r.effort) == "string" then effort = r.effort end
  if effort == nil and r.enabled == false then effort = "none" end
  if type(effort) ~= "string" or effort == "" or effort == "default" then effort = nil end
  local levels, defaults = effort_levels(model_name)
  if effort == nil and r.enabled == true and defaults and defaults.default_enabled == false then
    effort = "medium"
  end
  if type(r.max_tokens) == "number" then
    return nil, invalid_request("model " .. model_name .. " takes no reasoning.max_tokens; use reasoning_effort")
  end
  if effort ~= nil and levels then
    if not levels[effort] then
      return nil, invalid_request("reasoning_effort " .. effort .. " is not supported by model " .. model_name
        .. "; supported: " .. table.concat(effort_list(levels), ", "))
    end
    effort = levels[effort]
  end
  return effort, nil
end

-- ---------------------------------------------------------------------------
-- Payload
-- ---------------------------------------------------------------------------

-- Groq rejects every property it does not know ("property 'x' is
-- unsupported"), so the payload is built from a whitelist. Router fields
-- without a Groq counterpart (verbosity, prompt_cache_key, Ollama options...)
-- are dropped; known ones Groq refuses for a model (logprobs, n>1, stop>4)
-- reach it and come back with its own message.
local PASS_FIELDS = {
  "tools", "tool_choice", "parallel_tool_calls", "stream_options", "max_completion_tokens",
  "temperature", "top_p", "n", "stop", "seed", "frequency_penalty", "presence_penalty",
  "logprobs", "top_logprobs", "response_format", "user", "service_tier",
}
local MESSAGE_FIELDS = { "role", "content", "name", "tool_calls", "tool_call_id" }

local function build_messages(messages)
  local out = {}
  for i, m in ipairs(messages or {}) do
    local msg = {}
    for _, k in ipairs(MESSAGE_FIELDS) do msg[k] = m[k] end
    -- Groq takes earlier chain-of-thought as `reasoning`; the router's
    -- `reasoning_content` and `refusal` are unsupported properties there.
    local reasoning = m.reasoning_content or m.reasoning
    if m.role == "assistant" and type(reasoning) == "string" and reasoning ~= "" then
      msg.reasoning = reasoning
    end
    out[i] = msg
  end
  return out
end

-- (payload, error)
local function build_payload(request, stream)
  local effort, err = reasoning_effort_for(request)
  if err then return nil, err end
  local payload = { model = request.model_name, messages = build_messages(request.messages), stream = stream }
  for _, k in ipairs(PASS_FIELDS) do payload[k] = request[k] end
  if payload.max_completion_tokens == nil then payload.max_completion_tokens = request.max_tokens end
  if payload.seed == nil then payload.seed = request.random_seed end
  if not stream then payload.stream_options = nil end
  payload.reasoning_effort = effort
  if type(request.reasoning) == "table" and request.reasoning.exclude == true then
    -- Hidden works on every reasoning model (gpt-oss rejects only "raw").
    payload.reasoning_format = "hidden"
  end
  return payload, nil
end

-- ---------------------------------------------------------------------------
-- Audio helpers
-- ---------------------------------------------------------------------------

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

local function le32(n)
  return string.char(n % 256, math.floor(n / 256) % 256, math.floor(n / 65536) % 256, math.floor(n / 16777216) % 256)
end

-- Orpheus streams its WAV and leaves the RIFF and data sizes at 0xFFFFFFFF;
-- strict decoders reject that, so the sizes are written from the real length.
local function fix_wav_sizes(raw)
  if #raw < 44 or raw:sub(1, 4) ~= "RIFF" or raw:sub(9, 12) ~= "WAVE" then return raw end
  local pos = 13
  while pos + 7 <= #raw do
    local id = raw:sub(pos, pos + 3)
    local b1, b2, b3, b4 = raw:byte(pos + 4, pos + 7)
    if id == "data" then
      local data_len = #raw - (pos + 7)
      return "RIFF" .. le32(#raw - 8) .. raw:sub(9, pos + 3) .. le32(data_len) .. raw:sub(pos + 8)
    end
    pos = pos + 8 + b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  end
  return raw
end

-- ---------------------------------------------------------------------------
-- Plugin
-- ---------------------------------------------------------------------------

local function has(list, value)
  for _, item in ipairs(list or {}) do
    if item == value then return true end
  end
  return false
end

local function bearer(key, ctype)
  local h = { ["Authorization"] = "Bearer " .. key }
  if ctype then h["Content-Type"] = ctype end
  return h
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
      return nil, { message = "no Groq keys configured", code = "invalid_request_error", status = 400 }
    end
    -- Every key until one answers: a dead first key must not hide the list
    -- the others can fetch. The error names each key that failed and why.
    local client = llm_router.http_client({})
    local page, failures, last_status = nil, {}, 502
    for _, cred in ipairs(creds) do
      local resp, err = pooled_request(ctx, client, {
        method = "GET",
        url = BASE_URL .. "/models",
        headers = bearer(api_key_of(cred.data)),
      })
      local reason = nil
      if err then
        reason = (type(err) == "table" and err.message) or tostring(err)
      elseif resp.status == 200 then
        local ok, decoded = pcall(json.decode, resp.body or "")
        if ok and type(decoded) == "table" then
          page = decoded
          break
        end
        reason = "Groq returned an unreadable model list"
      elseif resp.status == 401 then
        reason = "Groq rejected the key: " .. upstream_message(resp, "invalid API key")
        last_status = 401
      else
        reason = "Groq answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details")
        last_status = resp.status
      end
      failures[#failures + 1] = "key " .. key_name(cred) .. ": " .. reason
    end
    if not page then
      return nil, {
        message = table.concat(failures, "; "),
        code = last_status == 401 and "authentication_error" or "server_error",
        status = last_status,
      }
    end

    local infos = {}
    for _, m in ipairs(page.data or {}) do
      local input = m.input_modalities or {}
      local output = m.output_modalities or {}
      local name = (m.name and m.name ~= "") and m.name or m.id

      if m.active and has(input, "audio")
          and (has(output, "text") or has(output, "transcription")) then
        local endpoints = { "audio/transcriptions" }
        -- Groq documents translation only for whisper-large-v3, not Turbo.
        if m.id == "whisper-large-v3" then
          table.insert(endpoints, "audio/translations")
        end
        table.insert(infos, {
          name = m.id,
          display_name = name,
          input_modalities = input,
          output_modalities = output,
          endpoints = endpoints,
        })
      elseif m.active and has(input, "text")
          and (has(output, "speech") or has(output, "audio")) then
        table.insert(infos, {
          name = m.id,
          display_name = name,
          input_modalities = input,
          output_modalities = output,
          endpoints = { "audio/speech" },
        })
      elseif m.active and has(input, "text") and has(output, "text") then
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
          reasoning = reasoning_info(m.id)
        end
        table.insert(infos, {
          name = m.id,
          display_name = name,
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
    local resp, err = pooled_request(ctx, client, { method = "GET", url = BASE_URL .. "/models", headers = bearer(key) })
    if err then
      local message = "request failed"
      if type(err) == "table" and type(err.message) == "string" then message = err.message end
      return { status = "unknown", message = message }
    end
    if resp.status == 200 then return { status = "healthy" } end
    if resp.status == 401 then
      return { status = "unhealthy", message = "Groq rejected the key: " .. upstream_message(resp, "invalid API key") }
    end
    return { status = "unknown", message = "Groq answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details") }
  end,

  -- Model liveness with zero generation spend, through the selected pool.
  -- Chat and Whisper: per-model metadata GET proves existence. Speech: an
  -- empty-input request is rejected before generation, with
  -- model_terms_required when the key's org has not accepted the terms and
  -- invalid_input when it has. Healthy wins at once, quota outranks unknown,
  -- unhealthy needs every tried key to agree. Probes never park or disable.
  check_model = function(ctx, model_ref)
    local endpoint = model_ref.endpoint or "chat"
    -- Vision-capable chat models still serve chat/completions upstream;
    -- vision is a dashboard-side input distinction, not a separate API.
    local speech = endpoint == "speech" or endpoint == "audio/speech"
    if not speech and endpoint ~= "chat" and endpoint ~= "vision" and endpoint ~= "transcription"
        and endpoint ~= "translation" and endpoint ~= "audio/transcriptions" and endpoint ~= "audio/translations" then
      return { status = "unknown", message = "endpoint not supported: " .. tostring(endpoint) }
    end
    local model_name = model_ref.model_name or ""
    if model_name == "" then return { status = "unknown", message = "model name is required" } end
    local creds = llm_router.credentials.list()
    if #creds == 0 then return { status = "unknown", message = "no credentials available" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local missing, terms, tried = 0, 0, 0
    local seen_quota = false
    local seen_unknown = nil
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key == "" then
        seen_unknown = seen_unknown or "api_key is required"
      else
        tried = tried + 1
        llm_router.log("debug", "probing model", { cred_id = cred.id, model = model_ref.model, endpoint = endpoint })
        local request
        if speech then
          request = {
            method = "POST",
            url = BASE_URL .. "/audio/speech",
            headers = bearer(key, "application/json"),
            body = json.encode({ model = model_name, input = "", voice = "" }),
          }
        else
          request = { method = "GET", url = BASE_URL .. "/models/" .. model_name, headers = bearer(key) }
        end
        local resp, err = pooled_request(ctx, client, request)
        local code = (not err) and upstream_code(resp) or nil
        if err then
          local message = "request failed"
          if type(err) == "table" and type(err.message) == "string" then message = err.message end
          seen_unknown = seen_unknown or message
        elseif code == "model_terms_required" then
          terms = terms + 1
          set_cooldown(cred.id, model_name, TERMS_COOLDOWN_SECS, TERMS_REASON)
          seen_unknown = seen_unknown or "model terms not accepted"
        elseif resp.status == 200 or (speech and resp.status == 400) then
          clear_cooldown(cred.id, model_name)
          return { status = "healthy", response = "ok" }
        elseif resp.status == 404 then
          missing = missing + 1
          seen_unknown = seen_unknown or "model not found upstream"
        elseif resp.status == 429 or code == "blocked_api_access" then
          seen_quota = true
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
    if tried > 0 and terms + missing == tried and terms > 0 then
      return { status = "unhealthy", message = terms_error(model_name).message }
    end
    if seen_quota then
      return { status = "quota_exceeded", message = "quota exhausted" }
    end
    return { status = "unknown", message = seen_unknown or "health check inconclusive" }
  end,

  complete = function(ctx, request)
    local payload, perr = build_payload(request, false)
    if perr then return nil, perr end
    local body = json.encode(payload)
    local client = llm_router.http_client({})
    local resp, err = attempt(ctx, request.model_name, function(cred, key, proxy_url)
      local r, e = client:request({
        method = "POST",
        url = BASE_URL .. "/chat/completions",
        headers = bearer(key, "application/json"),
        body = body,
        proxy_url = proxy_url,
      })
      return classify(ctx, cred, request.model_name, r, e)
    end)
    if not resp then return nil, err end
    local ok, out = pcall(json.decode, resp.body or "")
    if not ok or type(out) ~= "table" then
      return nil, error_of("groq returned invalid JSON", "server_error", 502)
    end
    out.model = request.model
    return out
  end,

  complete_stream = function(ctx, request, emit)
    local payload, perr = build_payload(request, true)
    if perr then return nil, perr end
    local body = json.encode(payload)
    local client = llm_router.http_client({})
    local full_model = request.model
    local emitted = false
    local _, err = attempt(ctx, request.model_name, function(cred, key, proxy_url)
      local action, value = nil, nil
      local _, stream_err = client:stream({
        method = "POST",
        url = BASE_URL .. "/chat/completions",
        headers = bearer(key, "application/json"),
        body = body,
        proxy_url = proxy_url,
        on_response = function(r)
          if r.status == 200 then
            note_limits(cred, request.model_name, r)
            return nil
          end
          action, value = classify(ctx, cred, request.model_name, r, nil)
          return value
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
          emitted = true
        end,
      })
      if action ~= nil then return action, value end
      if stream_err == nil then return "ok", true end
      -- A stream that emitted already belongs to that attempt: surface
      -- instead of failing over mid-stream.
      if emitted then return "stop", stream_err end
      return classify(ctx, cred, request.model_name, nil, stream_err)
    end)
    if err then return nil, err end
  end,

  -- Speech-to-text: always fetch verbose_json upstream; the router renders
  -- the client's response_format (json/text/srt/vtt) from the normalized
  -- response, so segments must be present when the client asked for them.
  transcribe = function(ctx, request)
    local task = request.task or "transcribe"
    if task ~= "transcribe" and task ~= "translate" then
      return nil, invalid_request("unsupported audio task: " .. tostring(task))
    end
    if task == "translate" and request.language and request.language ~= "" and request.language ~= "en" then
      return nil, invalid_request("Groq audio translations only accept language=en")
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
    local resp, err = attempt(ctx, request.model_name, function(cred, key, proxy_url)
      local r, e = client:request({
        method = "POST",
        url = BASE_URL .. path,
        headers = bearer(key, ctype),
        body = body,
        proxy_url = proxy_url,
      })
      return classify(ctx, cred, request.model_name, r, e)
    end)
    if not resp then return nil, err end
    local ok, out = pcall(json.decode, resp.body or "")
    if not ok or type(out) ~= "table" then
      return nil, error_of("groq returned invalid transcription JSON", "server_error", 502)
    end
    return out
  end,

  -- Orpheus TTS only supports WAV, requires a model-specific voice, and limits
  -- input to 200 characters. English vocal directions are expressed inline.
  speech = function(ctx, request)
    local model = request.model_name or ""
    local voices = ORPHEUS_VOICES[model]
    if not voices then
      return nil, invalid_request("unsupported Groq speech model: " .. model)
    end
    local text = tostring(request.input or "")
    local voice = tostring(request.voice or "")
    if text == "" then return nil, invalid_request("input is required") end
    if voice == "" then return nil, invalid_request("voice is required for Groq Orpheus") end
    if not voices[voice] then
      local names = {}
      for v in pairs(voices) do names[#names + 1] = v end
      table.sort(names)
      return nil, invalid_request("voice " .. voice .. " is not supported by " .. model .. "; supported: " .. table.concat(names, ", "))
    end
    local format = request.response_format
    if format ~= nil and format ~= "" and format ~= "wav" then
      return nil, invalid_request("Groq Orpheus supports only response_format=wav")
    end
    if request.speed ~= nil and tonumber(request.speed) and tonumber(request.speed) ~= 1 then
      return nil, invalid_request("Groq Orpheus does not support the speed parameter")
    end
    if request.instructions and request.instructions ~= "" then
      if model ~= "canopylabs/orpheus-v1-english" then
        return nil, invalid_request("vocal directions are supported only by the English Orpheus model")
      end
      local direction = tostring(request.instructions):gsub("^%[", ""):gsub("%]$", "")
      text = "[" .. direction .. "] " .. text
    end
    if utf8_length(text) > 200 then
      return nil, invalid_request("Groq Orpheus input is limited to 200 characters")
    end

    local body = json.encode({ model = model, input = text, voice = voice, response_format = "wav" })
    local client = llm_router.http_client({ timeout_ms = 300000 })
    local resp, err = attempt(ctx, model, function(cred, key, proxy_url)
      local r, e = client:request({
        method = "POST",
        url = BASE_URL .. "/audio/speech",
        headers = bearer(key, "application/json"),
        body = body,
        proxy_url = proxy_url,
      })
      return classify(ctx, cred, model, r, e)
    end)
    if not resp then return nil, err end
    if type(resp.body) ~= "string" or #resp.body == 0 then
      return nil, error_of("groq returned empty audio", "server_error", 502)
    end
    return { audio_b64 = llm_router.base64_encode(fix_wav_sizes(resp.body)), format = "wav" }
  end,
})
