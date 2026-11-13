--- @plugin Z.ai
--- @author TheSlopMachine
--- @version 0.1.1
--- @plugin_api 1.5
--- @description (beta) Z.ai (Zhipu GLM, international platform): chat, thinking switch, tools, vision; GLM Flash models are free
--- @allow_host api.z.ai

local BASE_URL = "https://api.z.ai/api/paas/v4"

local COOLDOWN_SCOPE = "rate_limit"
local DEFAULT_COOLDOWN_SECS = 60
local MAX_COOLDOWN_SECS = 86400
-- A key without balance or access for one model stays out of that model for an hour.
local MODEL_ACCESS_PARK_SECS = 3600

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

-- Z.ai errors: {"error": {"code": "1305", "message": "..."}} (the code is a
-- string); the legacy flat {"code": 1305, "message": "..."} is read too.
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
  if e and e.code ~= nil then return tostring(e.code) end
  return nil
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

-- Limits and model access bind to the model, so a 429 or an access error
-- cools only the key+model pair; parking would hide the key from every model.
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

-- Z.ai business codes (seen live unless marked): 1000 auth failed, 1211
-- unknown model, 1214 bad input, 1210 bad parameter, 1305 overloaded (sent
-- with HTTP 429, but it is the backend's load, not the key's limit).
-- Documented: 1113 no balance, 1220 no access to the model, 1308/1310
-- usage quota.
local CODE_UNKNOWN_MODEL = { ["1211"] = true, ["1212"] = true }
local CODE_NO_BALANCE = { ["1113"] = true, ["1112"] = true, ["1309"] = true }
local CODE_NO_ACCESS = { ["1220"] = true, ["1221"] = true, ["1222"] = true }
local CODE_QUOTA = { ["1308"] = true, ["1310"] = true, ["1304"] = true }
local CODE_AUTH = { ["1000"] = true, ["1001"] = true, ["1002"] = true, ["1003"] = true, ["1004"] = true }

-- Actions: "ok", "proxy" (next exit, then next key), "cred" (next key),
-- "server" (next exit, never another key), "stop" (final error).
local function classify(ctx, cred, model_name, resp, err)
  if err ~= nil then
    llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
    return "proxy", err
  end
  local status = resp.status
  if status == 200 then return "ok", resp end
  local message = upstream_message(resp, "Z.ai answered " .. tostring(status))
  local code = upstream_code(resp)
  local who = "Z.ai key " .. key_name(cred) .. " on model " .. tostring(model_name)

  if status == 401 or (code and CODE_AUTH[code]) then
    bench(ctx, cred.id, "z.ai rejected the api key", 300)
    llm_router.log("warn", "api key rejected", { status = status, code = code, cred_id = cred.id })
    return "cred", error_of("Z.ai rejected the key " .. key_name(cred) .. ": " .. message, "authentication_error", 401)
  end
  if code and CODE_UNKNOWN_MODEL[code] then
    return "stop", error_of("Z.ai: " .. message .. " (model " .. tostring(model_name) .. ")", "not_found", 404)
  end
  -- Model-level access: the key is fine, this model is not available to it.
  local lowered = message:lower()
  if (code and CODE_NO_BALANCE[code]) or status == 402 or lowered:find("insufficient balance", 1, true)
      or lowered:find("no resource package", 1, true) then
    set_cooldown(cred.id, model_name, MODEL_ACCESS_PARK_SECS, "no balance for model")
    llm_router.log("warn", "no balance for model", { status = status, code = code, cred_id = cred.id, model = model_name })
    return "cred", error_of(who .. " has no balance or package: " .. message, "payment_required", 402)
  end
  if (code and CODE_NO_ACCESS[code]) or (status == 403 and code ~= nil) then
    set_cooldown(cred.id, model_name, MODEL_ACCESS_PARK_SECS, "no access to model")
    llm_router.log("warn", "model not available to key", { status = status, code = code, cred_id = cred.id, model = model_name })
    return "cred", error_of(who .. " has no access: " .. message, "model_unavailable", 403)
  end
  if status == 403 then
    llm_router.log("debug", "access denied", { status = 403, cred_id = cred.id })
    return "proxy", error_of("Z.ai refused the request: " .. message, "server_error", 502)
  end
  if code == "1305" then
    -- Overload is the backend's, not the key's: other keys would hit the same wall.
    llm_router.log("debug", "z.ai overloaded", { status = status, model = model_name })
    return "stop", error_of("Z.ai is overloaded on model " .. tostring(model_name) .. ": " .. message, "overloaded", 503, retry_headers(5))
  end
  if status == 429 then
    local headers = resp.headers or {}
    local wait = tonumber(headers["retry-after"]) or DEFAULT_COOLDOWN_SECS
    local quota = code and CODE_QUOTA[code] or false
    set_cooldown(cred.id, model_name, wait, quota and "usage quota" or "rate limited")
    llm_router.log("warn", quota and "z.ai quota exhausted" or "z.ai rate limited",
      { status = 429, code = code, cred_id = cred.id, model = model_name, wait = wait })
    return "cred", error_of(who .. ": " .. message, quota and "insufficient_quota" or "rate_limit", 429, retry_headers(wait))
  end
  if status == 404 then
    return "stop", error_of(message, "not_found", 404)
  end
  if status == 400 or status == 413 or status == 422 then
    llm_router.log("error", "request rejected", { status = status, code = code, cred_id = cred.id })
    return "stop", error_of(message, "invalid_request_error", status)
  end
  if status == 503 then
    local wait = tonumber((resp.headers or {})["retry-after"]) or 5
    return "stop", error_of(message, "server_error", 503, retry_headers(wait))
  end
  llm_router.log("debug", "unexpected status", { status = status, code = code, cred_id = cred.id })
  if status >= 500 then return "server", error_of(message, "server_error", status) end
  return "stop", error_of(message, "server_error", status)
end

-- Credential x exit loop shared by both generation paths.
local function attempt(ctx, model_name, send)
  local usable, total, min_wait = plan_credentials(model_name)
  if total == 0 then
    return nil, error_of("no Z.ai keys configured", "server_error", 503)
  end
  if #usable == 0 then
    return nil, error_of("every Z.ai key is cooling down on " .. tostring(model_name) .. " (rate limit, no balance or no access)",
      "rate_limit", 429, retry_headers(min_wait))
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
  local exhausted = last_err or error_of("all Z.ai keys exhausted", "server_error", 502)
  llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
  return nil, exhausted
end

-- ---------------------------------------------------------------------------
-- Reasoning
-- ---------------------------------------------------------------------------

-- Z.ai has a thinking switch, not an effort scale: thinking {type =
-- enabled|disabled} (checked live 2026-11-12 on glm-4.7-flash and
-- glm-4.5-flash: reasoning_effort is accepted and ignored there). Router
-- scale: none = disabled, max = enabled (on is the model's full thinking).
-- low/medium/high would be a lie. Without a setting the model default
-- applies (thinking on).
local EFFORTS = { max = "enabled", none = "disabled" }
local EFFORT_LIST = { "max", "none" }

-- (thinking object to send or nil, error)
local function thinking_for(request)
  local model_name = request.model_name or ""
  local effort = request.reasoning_effort
  local r = type(request.reasoning) == "table" and request.reasoning or {}
  if effort == nil and type(r.effort) == "string" then effort = r.effort end
  if effort == nil and r.enabled == false then effort = "none" end
  if effort == nil and r.enabled == true then effort = "max" end
  if type(effort) ~= "string" or effort == "" or effort == "default" then effort = nil end
  if type(r.max_tokens) == "number" then
    return nil, invalid_request("model " .. model_name .. " takes no reasoning.max_tokens; use reasoning_effort")
  end
  if effort == nil then return nil, nil end
  if not EFFORTS[effort] then
    return nil, invalid_request("reasoning_effort " .. effort .. " is not supported by model " .. model_name
      .. "; supported: " .. table.concat(EFFORT_LIST, ", "))
  end
  return { type = EFFORTS[effort] }, nil
end

-- reasoning.exclude: the model still reasons; its text is dropped here.
local function reasoning_excluded(request)
  return type(request.reasoning) == "table" and request.reasoning.exclude == true
end

-- ---------------------------------------------------------------------------
-- Payload
-- ---------------------------------------------------------------------------

-- Z.ai ignores most unknown properties but validates the ones it knows
-- (temperature range, stop count), so the payload is a whitelist of what its
-- docs list. Router fields without a counterpart (n, seed, penalties,
-- logprobs, verbosity, metadata...) are dropped.
local PASS_FIELDS = { "tools", "tool_choice", "temperature", "top_p", "response_format" }
local MESSAGE_FIELDS = { "role", "content", "name", "tool_calls", "tool_call_id" }

local function is_vision_only(model_name)
  -- Vision-family models take no response_format (docs: text models only).
  local lower = tostring(model_name):lower()
  return lower:find("^glm%-%d+%.?%d*v") ~= nil or lower:find("flash", 1, true) ~= nil and lower:find("^glm%-5") ~= nil
end

local function build_messages(messages)
  local out = {}
  for i, m in ipairs(messages or {}) do
    local msg = {}
    for _, k in ipairs(MESSAGE_FIELDS) do msg[k] = m[k] end
    -- Earlier chain-of-thought goes back as reasoning_content; `refusal` is dropped.
    local reasoning = m.reasoning_content or m.reasoning
    if m.role == "assistant" and type(reasoning) == "string" and reasoning ~= "" then
      msg.reasoning_content = reasoning
    end
    out[i] = msg
  end
  return out
end

-- (payload, error)
local function build_payload(request, stream)
  local thinking, err = thinking_for(request)
  if err then return nil, err end
  local payload = { model = request.model_name, messages = build_messages(request.messages), stream = stream }
  for _, k in ipairs(PASS_FIELDS) do payload[k] = request[k] end
  payload.max_tokens = request.max_completion_tokens or request.max_tokens
  if is_vision_only(request.model_name) then payload.response_format = nil end
  -- Z.ai takes at most 4 stop words, as an array.
  local stop = request.stop
  if type(stop) == "string" then stop = { stop } end
  if type(stop) == "table" then payload.stop = stop end
  if type(request.user) == "string" and #request.user >= 6 and #request.user <= 128 then
    payload.user_id = request.user
  end
  payload.thinking = thinking
  return payload, nil
end

-- ---------------------------------------------------------------------------
-- Model catalog
-- ---------------------------------------------------------------------------

-- GET /models lists only the models a key may call on the paid tiers and no
-- metadata; the free Flash models are missing from it. They are added when a
-- zero-cost probe (a request with an empty prompt, no generation) shows the
-- model exists. Limits below are the ones in docs.z.ai; 0 = not documented.
local FREE_MODELS = { "glm-4.7-flash", "glm-4.5-flash", "glm-4.6v-flash" }

local SPECS = {
  ["glm-4.7-flash"]  = { context = 200000, output = 131072 },
  ["glm-4.7"]        = { context = 200000, output = 131072 },
  ["glm-4.6"]        = { context = 200000, output = 131072 },
  ["glm-4.5-flash"]  = { context = 128000, output = 98304 },
  ["glm-4.5"]        = { context = 128000, output = 98304 },
  ["glm-4.5-air"]    = { context = 128000, output = 98304 },
  ["glm-4.6v-flash"] = { context = 128000, output = 0, vision = true },
  ["glm-5.3-flash"]  = { vision = true },
  ["glm-5.3-flashx"] = { vision = true },
  ["glm-5.3-flashx-legal"] = { vision = true },
}

local function model_info_of(id)
  local spec = SPECS[id] or {}
  local params = { "temperature", "top_p", "max_tokens", "stop", "tools", "tool_choice", "reasoning", "reasoning_effort" }
  if not is_vision_only(id) then params[#params + 1] = "response_format" end
  local input = { "text" }
  if spec.vision then input[#input + 1] = "image" end
  return {
    name = id,
    display_name = id,
    context_window = spec.context or 0,
    max_tokens = spec.output or 0,
    input_modalities = input,
    output_modalities = { "text" },
    supported_parameters = params,
    reasoning = { supported_efforts = EFFORT_LIST, default_enabled = true },
    endpoints = { "chat/completions" },
  }
end

-- Zero-cost existence probe: an empty prompt fails validation after the model
-- lookup. Unknown model -> 1211, known -> 1213 (prompt not received).
-- Returns true / false / nil (inconclusive).
local function model_exists(ctx, client, key, model_name)
  local resp, err = pooled_request(ctx, client, {
    method = "POST",
    url = BASE_URL .. "/chat/completions",
    headers = bearer(key, "application/json"),
    body = json.encode({ model = model_name, messages = { { role = "user", content = "" } } }),
  })
  if err or not resp then return nil end
  local code = upstream_code(resp)
  if code and CODE_UNKNOWN_MODEL[code] then return false end
  -- 1305 (overloaded) and 1302 (rate limit) also come after the lookup: the
  -- model exists, it is just busy (glm-4.6v-flash does this often).
  if code == "1213" or code == "1214" or code == "1305" or code == "1302" then return true end
  return nil
end

-- ---------------------------------------------------------------------------
-- Plugin
-- ---------------------------------------------------------------------------

llm_router.register("zai", {
  icon = "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",

  proxy_schema = {},

  locales = {
    ["en"] = {
      ["zai.subtitle"] = "Create an API key on the Z.ai platform. GLM Flash models are free.",
      ["zai.open_console"] = "Open Z.ai API keys",
      ["misc.api_key"] = "API Key",
      ["misc.save"] = "Save",
    },
    ["ru"] = {
      ["zai.subtitle"] = "Создайте API-ключ на платформе Z.ai. Модели GLM Flash бесплатны.",
      ["zai.open_console"] = "Открыть ключи Z.ai",
      ["misc.api_key"] = "API-ключ",
      ["misc.save"] = "Сохранить",
    },
  },

  credential_schema = {
    {
      type = "section",
      title = "Z.ai",
      subtitle = "zai.subtitle",
      content = {
        { type = "link", text = "zai.open_console", url = "https://z.ai/manage-apikey/apikey-list" },
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
    if #key < 10 then return false, { message = "api_key: minimum 10 characters", code = "invalid_request_error" } end
    if key:find("%s") then return false, { message = "api_key must not contain whitespace", code = "invalid_request_error" } end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, { message = "no Z.ai keys configured", code = "invalid_request_error", status = 400 }
    end
    -- Every key until one answers; the error names each key that failed and why.
    local client = llm_router.http_client({ timeout_ms = 20000 })
    local ids, good_key, failures, last_status = nil, nil, {}, 502
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      local resp, err = pooled_request(ctx, client, { method = "GET", url = BASE_URL .. "/models", headers = bearer(key) })
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
          good_key = key
          break
        end
        reason = "Z.ai returned an unreadable model list"
      elseif resp.status == 401 then
        reason = "Z.ai rejected the key: " .. upstream_message(resp, "invalid API key")
        last_status = 401
      else
        reason = "Z.ai answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details")
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
    local seen = {}
    for _, id in ipairs(ids) do seen[id] = true end
    for _, id in ipairs(FREE_MODELS) do
      if not seen[id] and model_exists(ctx, client, good_key, id) == true then
        ids[#ids + 1] = id
        seen[id] = true
      end
    end
    local infos = {}
    for _, id in ipairs(ids) do infos[#infos + 1] = model_info_of(id) end
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
      return { status = "unhealthy", message = "Z.ai rejected the key: " .. upstream_message(resp, "invalid API key") }
    end
    return { status = "unknown", message = "Z.ai answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details") }
  end,

  -- Zero generation spend: a request with an empty prompt is validated after
  -- the model lookup, so it tells a missing model from an existing one.
  -- Whether the key has balance for a paid model shows only at generation.
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
    local missing, tried, seen_unknown = 0, 0, nil
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key == "" then
        seen_unknown = seen_unknown or "api_key is required"
      else
        tried = tried + 1
        local exists = model_exists(ctx, client, key, model_name)
        if exists == true then
          return { status = "healthy", response = "ok" }
        elseif exists == false then
          missing = missing + 1
        else
          seen_unknown = seen_unknown or "Z.ai gave no verdict on the model"
        end
      end
    end
    if tried > 0 and missing == tried then
      return { status = "unhealthy", message = "model not found upstream" }
    end
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
      return nil, error_of("Z.ai returned invalid JSON", "server_error", 502)
    end
    if reasoning_excluded(request) then
      for _, choice in ipairs(out.choices or {}) do
        if type(choice.message) == "table" then
          choice.message.reasoning_content = nil
          choice.message.reasoning = nil
        end
      end
    end
    out.web_search = nil
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
          if line:sub(1, 5) ~= "data:" then return end
          local data = line:sub(6):match("^%s*(.-)%s*$")
          if data == "[DONE]" or data == "" then return end
          local ok, chunk = pcall(json.decode, data)
          if not ok or type(chunk) ~= "table" then return end
          local has_choices = type(chunk.choices) == "table" and #chunk.choices > 0
          if hide_reasoning and has_choices then
            for _, choice in ipairs(chunk.choices) do
              if type(choice.delta) == "table" then
                choice.delta.reasoning_content = nil
                choice.delta.reasoning = nil
              end
            end
          end
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
})
