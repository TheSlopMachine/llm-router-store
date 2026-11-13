--- @plugin Cerebras
--- @author TheSlopMachine
--- @version 1.0.1
--- @plugin_api 1.5
--- @description Cerebras Inference: fast chat, reasoning, tools, structured outputs, vision
--- @allow_host api.cerebras.ai

local BASE_URL = "https://api.cerebras.ai/v1"
-- Keyless catalog with context, output cap, capabilities and parameters per model.
local PUBLIC_MODELS_URL = "https://api.cerebras.ai/public/v1/models"

local COOLDOWN_SCOPE = "rate_limit"
local DEFAULT_COOLDOWN_SECS = 60
local MAX_COOLDOWN_SECS = 86400
-- 402: the account ran out of credit. Parked for a day: topping up needs a person.
local PAYMENT_PARK_SECS = 86400

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

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

local function bearer(key, ctype)
  local h = { ["Authorization"] = "Bearer " .. key }
  if ctype then h["Content-Type"] = ctype end
  return h
end

-- Cerebras errors are flat ({message, type, param, code}); the OpenAI
-- envelope ({error: {...}}) is read too.
local function upstream_error(resp)
  local body = resp and resp.body
  if type(body) ~= "string" or body == "" then return nil end
  local ok, parsed = pcall(json.decode, body)
  if not ok or type(parsed) ~= "table" then return nil end
  if type(parsed.error) == "table" then return parsed.error end
  if type(parsed.message) == "string" then return parsed end
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

local function bench(ctx, cred_id, reason, wait_secs)
  local cfg = ctx.provider_config
  if type(cfg) == "table" and cfg.disable_failed_credentials == true then
    if pcall(llm_router.credentials.disable, cred_id, reason) then return end
  end
  pcall(llm_router.credentials.park, cred_id, wait_secs, reason)
end

-- ---------------------------------------------------------------------------
-- Per key+model cooldowns
-- ---------------------------------------------------------------------------

-- Cerebras limits bind to the organization and the model (gpt-oss and qwen
-- carry separate x-ratelimit-* buckets), so a 429 cools only the key+model
-- pair; parking would hide the key from every model.
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

local function cooldown_remaining(cred_id, model_name)
  local ok, value = pcall(llm_router.storage.get, COOLDOWN_SCOPE, pair_key(cred_id, model_name))
  if not ok or type(value) ~= "table" then return nil end
  local remaining = (tonumber(value["until"]) or 0) - os.time()
  if remaining > 0 then return remaining end
  return nil
end

-- (usable, total, min_wait)
local function plan_credentials(model_name)
  local usable, total, min_wait = {}, 0, nil
  for _, cred in ipairs(llm_router.credentials.list()) do
    total = total + 1
    local remaining = cooldown_remaining(cred.id, model_name)
    if remaining ~= nil then
      min_wait = min_wait and math.min(min_wait, remaining) or remaining
    else
      usable[#usable + 1] = cred
    end
  end
  return usable, total, min_wait
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
  local res = llm_router.proxies.require({ pool = pool, limit = limit or 3, fallback = "direct" })
  if not res or #res.proxies == 0 then
    return { {} }
  end
  return res.proxies
end

-- One request through the selected pool, for every call that is not
-- generation. A 403, a 5xx or a transport fault on one exit moves on to the
-- next; a direct pool yields one direct attempt.
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

-- Actions: "ok", "proxy" (next exit, then next key), "cred" (next key),
-- "server" (next exit, never another key), "stop" (final error).
local function classify(ctx, cred, model_name, resp, err)
  if err ~= nil then
    llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
    return "proxy", err
  end
  local status = resp.status
  if status == 200 then return "ok", resp end
  local message = upstream_message(resp, "Cerebras answered " .. tostring(status))
  if status == 401 then
    bench(ctx, cred.id, "cerebras rejected the api key", 300)
    llm_router.log("warn", "api key rejected", { status = 401, cred_id = cred.id })
    return "cred", error_of("Cerebras rejected the key " .. key_name(cred) .. ": " .. message, "authentication_error", 401)
  end
  if status == 402 then
    pcall(llm_router.credentials.park, cred.id, PAYMENT_PARK_SECS, "cerebras credit exhausted")
    llm_router.log("warn", "credit exhausted", { status = 402, cred_id = cred.id })
    return "cred", error_of("Cerebras key " .. key_name(cred) .. " has no credit left: " .. message, "payment_required", 402)
  end
  if status == 403 then
    llm_router.log("debug", "access denied", { status = 403, cred_id = cred.id })
    return "proxy", error_of("Cerebras refused the request: " .. message, "server_error", 502)
  end
  if status == 429 then
    local headers = resp.headers or {}
    local wait = tonumber(headers["retry-after"]) or DEFAULT_COOLDOWN_SECS
    local daily = message:lower():find("day", 1, true) ~= nil
    set_cooldown(cred.id, model_name, wait, daily and "daily quota" or "rate limited")
    llm_router.log("warn", daily and "cerebras daily quota exhausted" or "cerebras rate limited",
      { status = 429, cred_id = cred.id, model = model_name, wait = wait })
    return "cred", error_of(message, daily and "insufficient_quota" or "rate_limit", 429, retry_headers(wait))
  end
  if status == 404 then
    return "stop", error_of(message, "not_found", 404)
  end
  if status == 400 or status == 413 or status == 422 then
    llm_router.log("error", "request rejected", { status = status, code = upstream_code(resp), cred_id = cred.id })
    return "stop", error_of(message, "invalid_request_error", status)
  end
  if status == 503 then
    -- Overload is the backend's, not the key's.
    local wait = tonumber((resp.headers or {})["retry-after"]) or 5
    return "stop", error_of(message, "server_error", 503, retry_headers(wait))
  end
  llm_router.log("debug", "unexpected status", { status = status, cred_id = cred.id })
  if status >= 500 then return "server", error_of(message, "server_error", status) end
  return "stop", error_of(message, "server_error", status)
end

-- Credential x exit loop shared by both generation paths.
local function attempt(ctx, model_name, send)
  local usable, total, min_wait = plan_credentials(model_name)
  if total == 0 then
    return nil, error_of("no Cerebras keys configured", "server_error", 503)
  end
  if #usable == 0 then
    return nil, error_of("every Cerebras key is rate limited on " .. tostring(model_name), "rate_limit", 429, retry_headers(min_wait))
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
  local left, _, wait = plan_credentials(model_name)
  if #left == 0 and last_err and (last_err.code == "rate_limit" or last_err.code == "insufficient_quota") then
    last_err.headers = retry_headers(wait) or last_err.headers
  end
  local exhausted = last_err or error_of("all Cerebras keys exhausted", "server_error", 502)
  llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
  return nil, exhausted
end

-- ---------------------------------------------------------------------------
-- Reasoning
-- ---------------------------------------------------------------------------

-- Router scale none/low/medium/high/max onto Cerebras reasoning_effort
-- (checked live 2026-11-12). "default" means the model default: accepted,
-- never advertised, sent as nothing.
--   gpt-oss:  high, medium, low   (reasoning always runs; default medium)
--   qwen-3.x: high, medium, low, none (default high)
local EFFORT_ORDER = { "max", "high", "medium", "low", "none" }

local function effort_levels(model_name)
  local lower = tostring(model_name):lower()
  if lower:find("gpt%-oss") then
    return { high = "high", medium = "medium", low = "low" }, { default_effort = "medium", default_enabled = true, mandatory = true }
  end
  if lower:find("^qwen%-3") then
    return { high = "high", medium = "medium", low = "low", none = "none" }, { default_effort = "high", default_enabled = true }
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

-- (reasoning_effort to send or nil, error)
local function reasoning_effort_for(request)
  local model_name = request.model_name or ""
  local effort = request.reasoning_effort
  local r = type(request.reasoning) == "table" and request.reasoning or {}
  if effort == nil and type(r.effort) == "string" then effort = r.effort end
  if effort == nil and r.enabled == false then effort = "none" end
  if type(effort) ~= "string" or effort == "" or effort == "default" then effort = nil end
  if type(r.max_tokens) == "number" then
    return nil, invalid_request("model " .. model_name .. " takes no reasoning.max_tokens; use reasoning_effort")
  end
  local levels = effort_levels(model_name)
  if effort ~= nil and levels then
    if not levels[effort] then
      return nil, invalid_request("reasoning_effort " .. effort .. " is not supported by model " .. model_name
        .. "; supported: " .. table.concat(effort_list(levels), ", "))
    end
    effort = levels[effort]
  end
  return effort, nil
end

-- reasoning.exclude: the model still reasons; its text is dropped here
-- (qwen rejects reasoning_format "hidden", so the plugin strips it itself).
local function reasoning_excluded(request)
  return type(request.reasoning) == "table" and request.reasoning.exclude == true
end

-- ---------------------------------------------------------------------------
-- Payload
-- ---------------------------------------------------------------------------

-- Cerebras rejects every property it does not know ("property 'x' is
-- unsupported"), so the payload is built from a whitelist. Router fields
-- without a Cerebras counterpart (verbosity, metadata, Ollama options...) are
-- dropped; known ones Cerebras refuses (n>1, stop>4) reach it and come back
-- with its own message.
local PASS_FIELDS = {
  "tools", "tool_choice", "parallel_tool_calls", "stream_options", "max_completion_tokens",
  "temperature", "top_p", "n", "stop", "seed", "frequency_penalty", "presence_penalty",
  "logprobs", "top_logprobs", "response_format", "user", "prompt_cache_key",
}
local MESSAGE_FIELDS = { "role", "content", "name", "tool_calls", "tool_call_id" }

local function build_messages(messages)
  local out = {}
  for i, m in ipairs(messages or {}) do
    local msg = {}
    for _, k in ipairs(MESSAGE_FIELDS) do msg[k] = m[k] end
    -- Earlier chain-of-thought goes back as `reasoning`; the router's
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
  -- max_tokens and max_completion_tokens are aliases there; both at once is a 400.
  if payload.max_completion_tokens == nil then payload.max_completion_tokens = request.max_tokens end
  if payload.seed == nil then payload.seed = request.random_seed end
  if not stream then payload.stream_options = nil end
  payload.reasoning_effort = effort
  return payload, nil
end

-- ---------------------------------------------------------------------------
-- Model catalog
-- ---------------------------------------------------------------------------

-- The keyed /v1/models says which models a key may use (ids only); the
-- keyless public catalog describes them. A model the public catalog does not
-- describe is still listed, with what the router can say about it.
local function catalog_by_id(ctx, client)
  local resp, err = pooled_request(ctx, client, { method = "GET", url = PUBLIC_MODELS_URL })
  if err or resp.status ~= 200 then
    llm_router.log("warn", "public catalog unavailable", { status = resp and resp.status or nil })
    return {}
  end
  local ok, page = pcall(json.decode, resp.body or "")
  local by_id = {}
  if ok and type(page) == "table" then
    for _, m in ipairs(page.data or {}) do
      if type(m) == "table" and type(m.id) == "string" then by_id[m.id] = m end
    end
  end
  return by_id
end

local function model_info_of(id, meta)
  meta = meta or {}
  local caps = type(meta.capabilities) == "table" and meta.capabilities or {}
  local limits = type(meta.limits) == "table" and meta.limits or {}
  local params = {}
  local sampling = type(meta.supported_parameters) == "table" and meta.supported_parameters or {}
  for _, p in ipairs({ "temperature", "top_p", "seed", "stop", "max_completion_tokens", "logprobs", "top_logprobs",
                       "frequency_penalty", "presence_penalty" }) do
    if sampling[p] == true then params[#params + 1] = p end
  end
  if caps.tools == true or caps.function_calling == true then
    params[#params + 1] = "tools"
    params[#params + 1] = "tool_choice"
  end
  if caps.parallel_tool_calls == true then params[#params + 1] = "parallel_tool_calls" end
  if caps.response_format == true or caps.json_mode == true then params[#params + 1] = "response_format" end
  if caps.structured_outputs == true then params[#params + 1] = "structured_outputs" end
  local reasoning = nil
  if caps.reasoning == true or effort_levels(id) ~= nil then
    params[#params + 1] = "reasoning"
    params[#params + 1] = "reasoning_effort"
    reasoning = reasoning_info(id)
  end
  local input = { "text" }
  if caps.vision == true then input[#input + 1] = "image" end
  return {
    name = id,
    display_name = (type(meta.name) == "string" and meta.name ~= "") and meta.name or id,
    description = type(meta.description) == "string" and meta.description or nil,
    context_window = tonumber(limits.max_context_length) or 0,
    max_tokens = tonumber(limits.max_completion_tokens) or 0,
    input_modalities = input,
    output_modalities = { "text" },
    supported_parameters = params,
    reasoning = reasoning,
    endpoints = { "chat/completions" },
  }
end

-- ---------------------------------------------------------------------------
-- Plugin
-- ---------------------------------------------------------------------------

llm_router.register("cerebras", {
  icon = "https://cdn.sanity.io/images/e4qjo92p/production/e7a55ae5ab7e2c4fdfd4e66a51f628d1f2f44207-967x967.png?w=128&h=128&fit=max&auto=format",

  proxy_schema = {},

  locales = {
    ["en"] = {
      ["cerebras.subtitle"] = "Get your API key from the Cerebras Cloud console.",
      ["cerebras.open_console"] = "Open Cerebras Cloud",
      ["misc.api_key"] = "API Key",
      ["misc.save"] = "Save",
    },
    ["ru"] = {
      ["cerebras.subtitle"] = "Получите API-ключ в консоли Cerebras Cloud.",
      ["cerebras.open_console"] = "Открыть Cerebras Cloud",
      ["misc.api_key"] = "API-ключ",
      ["misc.save"] = "Сохранить",
    },
  },

  credential_schema = {
    {
      type = "section",
      title = "Cerebras",
      subtitle = "cerebras.subtitle",
      content = {
        { type = "link", text = "cerebras.open_console", url = "https://cloud.cerebras.ai/platform" },
        { type = "secret", name = "api_key", label = "misc.api_key", required = true },
        { type = "button", text = "misc.save", form_action = "submit" },
      },
    },
  },

  validate_credentials = function(data)
    local key = type(data) == "table" and data.api_key or nil
    if type(key) ~= "string" or key == "" then
      return false, { message = "api_key is required", code = "invalid_request_error" }
    end
    if #key < 20 then return false, { message = "api_key: minimum 20 characters", code = "invalid_request_error" } end
    if key:find("%s") then return false, { message = "api_key must not contain whitespace", code = "invalid_request_error" } end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, { message = "no Cerebras keys configured", code = "invalid_request_error", status = 400 }
    end
    -- Every key until one answers; the error names each key that failed and why.
    local client = llm_router.http_client({ timeout_ms = 20000 })
    local ids, failures, last_status = nil, {}, 502
    for _, cred in ipairs(creds) do
      local resp, err = pooled_request(ctx, client, { method = "GET", url = BASE_URL .. "/models", headers = bearer(api_key_of(cred.data)) })
      local reason = nil
      if err then
        reason = "request failed: " .. ((type(err) == "table" and err.message) or tostring(err))
      elseif resp.status == 200 then
        local ok, page = pcall(json.decode, resp.body or "")
        if ok and type(page) == "table" and type(page.data) == "table" then
          ids = {}
          for _, m in ipairs(page.data) do
            if type(m) == "table" and type(m.id) == "string" then ids[#ids + 1] = m.id end
          end
          break
        end
        reason = "Cerebras returned an unreadable model list"
      elseif resp.status == 401 then
        reason = "Cerebras rejected the key: " .. upstream_message(resp, "invalid API key")
        last_status = 401
      else
        reason = "Cerebras answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details")
        last_status = resp.status
      end
      failures[#failures + 1] = "key " .. key_name(cred) .. ": " .. reason
    end
    if not ids then
      return nil, {
        message = table.concat(failures, "; "),
        code = last_status == 401 and "authentication_error" or "server_error",
        status = last_status,
      }
    end
    local meta = catalog_by_id(ctx, client)
    local infos = {}
    for _, id in ipairs(ids) do
      local m = meta[id]
      if not (m and m.deprecated == true) then infos[#infos + 1] = model_info_of(id, m) end
    end
    table.sort(infos, function(a, b) return a.name < b.name end)
    return infos
  end,

  -- Account liveness: the keyed /models listing answers 200 for live keys
  -- and 401 for dead ones. Only 401 verdicts unhealthy.
  check_health = function(ctx, credential)
    local key = api_key_of(credential and credential.data)
    if key == "" then return { status = "unhealthy", message = "api_key is required" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local resp, err = pooled_request(ctx, client, { method = "GET", url = BASE_URL .. "/models", headers = bearer(key) })
    if err then
      return { status = "unknown", message = (type(err) == "table" and err.message) or "request failed" }
    end
    if resp.status == 200 then return { status = "healthy" } end
    if resp.status == 401 then
      return { status = "unhealthy", message = "Cerebras rejected the key: " .. upstream_message(resp, "invalid API key") }
    end
    return { status = "unknown", message = "Cerebras answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details") }
  end,

  -- Model liveness with zero generation spend: GET /models/{id} per key.
  -- Healthy wins at once; unhealthy needs every tried key to miss the model.
  check_model = function(ctx, model_ref)
    local endpoint = model_ref.endpoint or "chat"
    if endpoint ~= "chat" and endpoint ~= "vision" then
      return { status = "unknown", message = "endpoint not supported: " .. tostring(endpoint) }
    end
    local model_name = model_ref.model_name or ""
    if model_name == "" then return { status = "unknown", message = "model name is required" } end
    local creds = llm_router.credentials.list()
    if #creds == 0 then return { status = "unknown", message = "no credentials available" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local missing, tried, seen_quota, seen_unknown = 0, 0, false, nil
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key == "" then
        seen_unknown = seen_unknown or "api_key is required"
      else
        tried = tried + 1
        local resp, err = pooled_request(ctx, client, { method = "GET", url = BASE_URL .. "/models/" .. model_name, headers = bearer(key) })
        if err then
          seen_unknown = seen_unknown or ((type(err) == "table" and err.message) or "request failed")
        elseif resp.status == 200 then
          return { status = "healthy", response = "ok" }
        elseif resp.status == 404 then
          missing = missing + 1
        elseif resp.status == 429 or resp.status == 402 then
          seen_quota = true
        else
          seen_unknown = seen_unknown or ("Cerebras answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details"))
        end
      end
    end
    if tried > 0 and missing == tried then
      return { status = "unhealthy", message = "model not found upstream" }
    end
    if seen_quota then return { status = "quota_exceeded", message = "quota exhausted" } end
    return { status = "unknown", message = seen_unknown or "health check inconclusive" }
  end,

  complete = function(ctx, request)
    local payload, perr = build_payload(request, false)
    if perr then return nil, perr end
    local body = json.encode(payload)
    local client = llm_router.http_client({ timeout_ms = 300000 })
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
      return nil, error_of("Cerebras returned invalid JSON", "server_error", 502)
    end
    if reasoning_excluded(request) then
      for _, choice in ipairs(out.choices or {}) do
        if type(choice.message) == "table" then choice.message.reasoning = nil end
      end
    end
    out.time_info = nil
    out.model = request.model
    return out
  end,

  complete_stream = function(ctx, request, emit)
    local payload, perr = build_payload(request, true)
    if perr then return nil, perr end
    local body = json.encode(payload)
    local client = llm_router.http_client({ timeout_ms = 300000 })
    local full_model = request.model
    local hide_reasoning = reasoning_excluded(request)
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
          if r.status == 200 then return nil end
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
          if hide_reasoning and has_choices then
            for _, choice in ipairs(chunk.choices) do
              if type(choice.delta) == "table" then choice.delta.reasoning = nil end
            end
          end
          if not has_choices and chunk.usage == nil then return end
          chunk.time_info = nil
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
})
