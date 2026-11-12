--- @plugin Alibaba Model Studio
--- @author TheSlopMachine
--- @version 0.1.0
--- @plugin_api 1.5
--- @description (beta) Alibaba Cloud Model Studio (DashScope, OpenAI-compatible mode): Qwen, DeepSeek, GLM and Kimi chat with thinking control, tools, image input, embeddings
--- @allow_host dashscope-intl.aliyuncs.com
--- @allow_host dashscope.aliyuncs.com

-- Endpoint choice. Model Studio keys, model lists and quotas are bound to a
-- region, and every region has its own host, so the host is derived from a
-- per-credential "region" setting (default: Singapore, the international
-- site with the free quota). The shared DashScope hosts are used rather than
-- the workspace-dedicated `<workspace>.<region>.maas.aliyuncs.com` ones: a
-- workspace host cannot be listed in @allow_host (no wildcards), and the same
-- key answers on both. Regions that exist only behind workspace hosts
-- (Frankfurt, Tokyo, Virginia, Hong Kong) are therefore not offered.
-- Only Singapore was exercised against a live key; Beijing is the same API on
-- the China-site host.
local REGIONS = {
  singapore = "https://dashscope-intl.aliyuncs.com",
  beijing = "https://dashscope.aliyuncs.com",
}
local DEFAULT_REGION = "singapore"

local COOLDOWN_SCOPE = "rate_limit"
local DEFAULT_COOLDOWN_SECS = 60
local MAX_COOLDOWN_SECS = 86400
-- Model-level refusals (model not activated for the key, access denied) are
-- re-tried after an hour; an exhausted free tier after a day.
local ACCESS_COOLDOWN_SECS = 3600
local QUOTA_COOLDOWN_SECS = 86400
-- Arrears on the account: parked for a day, topping up needs a person.
local PAYMENT_PARK_SECS = 86400

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

local function api_key_of(data)
  if type(data) ~= "table" or type(data.api_key) ~= "string" then return "" end
  return (data.api_key:match("^%s*(.-)%s*$"))
end

-- Host root (no path) of the credential's region.
local function host_of(cred)
  local data = type(cred) == "table" and cred.data or nil
  local region = type(data) == "table" and data.region or nil
  if type(region) == "string" and REGIONS[region] then return REGIONS[region] end
  return REGIONS[DEFAULT_REGION]
end

local function base_of(cred)
  return host_of(cred) .. "/compatible-mode/v1"
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

-- Two error shapes: {error = {message, type, param, code}} and the flat
-- {code, message, request_id} (model not activated, gateway errors).
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
  return e and type(e.code) == "string" and e.code or ""
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

-- Rate limits, free-tier quotas and model activation are per model, so a
-- failure cools only the key+model pair; parking would hide the key from
-- every model.
local function pair_key(cred_id, model_name)
  return tostring(cred_id) .. ":" .. tostring(model_name)
end

local function set_cooldown(cred_id, model_name, secs, reason, code, status)
  secs = math.ceil(tonumber(secs) or DEFAULT_COOLDOWN_SECS)
  if secs < 1 then secs = 1 end
  if secs > MAX_COOLDOWN_SECS then secs = MAX_COOLDOWN_SECS end
  local now = os.time()
  pcall(llm_router.storage.set, COOLDOWN_SCOPE, pair_key(cred_id, model_name),
    { ["until"] = now + secs, reason = reason or "rate limited", code = code or "rate_limit", status = status or 429 },
    { ttl = secs })
end

-- (remaining seconds or nil, stored entry)
local function cooldown_of(cred_id, model_name)
  local ok, value = pcall(llm_router.storage.get, COOLDOWN_SCOPE, pair_key(cred_id, model_name))
  if not ok or type(value) ~= "table" then return nil end
  local remaining = (tonumber(value["until"]) or 0) - os.time()
  if remaining > 0 then return remaining, value end
  return nil
end

-- (usable, total, min_wait, entry that frees first)
local function plan_credentials(model_name)
  local usable, total, min_wait, min_entry = {}, 0, nil, nil
  for _, cred in ipairs(llm_router.credentials.list()) do
    total = total + 1
    local remaining, entry = cooldown_of(cred.id, model_name)
    if remaining ~= nil then
      if min_wait == nil or remaining < min_wait then
        min_wait, min_entry = remaining, entry
      end
    else
      usable[#usable + 1] = cred
    end
  end
  return usable, total, min_wait, min_entry
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

local function contains(text, ...)
  local lower = tostring(text or ""):lower()
  for _, needle in ipairs({ ... }) do
    if lower:find(needle, 1, true) then return true end
  end
  return false
end

-- Account in arrears: every model is closed until a person tops up.
local function is_arrears(code, message)
  return contains(code, "arrearage") or contains(message, "good standing", "overdue", "arrear")
end

-- Free tier of one model used up (the "use free quota only" mode is on).
local function is_free_tier_gone(code, message)
  return contains(code, "freetier") or contains(message, "free tier", "free quota")
end

-- The key may not use this model (not activated, not authorised).
local function is_model_access(code, message)
  return contains(code, "accessdenied", "access_denied", "unpurchased", "notactivated")
    or contains(message, "not activated", "access denied", "no permission", "not authorized", "do not have access")
end

-- Actions: "ok", "proxy" (next exit, then next key), "cred" (next key),
-- "server" (next exit, never another key), "stop" (final error).
local function classify(ctx, cred, model_name, resp, err)
  if err ~= nil then
    llm_router.log("debug", "transport error", { reason = tostring(err.message or "transport error"), code = err.code, cred_id = cred.id })
    return "proxy", err
  end
  local status = resp.status
  if status == 200 then return "ok", resp end
  local message = upstream_message(resp, "Alibaba Model Studio answered " .. tostring(status))
  local code = upstream_code(resp)
  local who = "Alibaba Model Studio key " .. key_name(cred)

  if status == 401 then
    bench(ctx, cred.id, "alibaba rejected the api key", 300)
    llm_router.log("warn", "api key rejected", { status = 401, cred_id = cred.id })
    return "cred", error_of("Alibaba Model Studio rejected the key " .. key_name(cred) .. ": " .. message, "authentication_error", 401)
  end
  if status == 402 or ((status == 400 or status == 403) and is_arrears(code, message)) then
    pcall(llm_router.credentials.park, cred.id, PAYMENT_PARK_SECS, "alibaba account in arrears")
    llm_router.log("warn", "account in arrears", { status = status, cred_id = cred.id })
    return "cred", error_of(who .. " has no credit left: " .. message, "payment_required", 402)
  end
  if (status == 400 or status == 403 or status == 429) and is_free_tier_gone(code, message) then
    local text = who .. " has used up the free quota of " .. tostring(model_name) .. ": " .. message
    set_cooldown(cred.id, model_name, QUOTA_COOLDOWN_SECS, text, "insufficient_quota", 429)
    llm_router.log("warn", "free quota exhausted", { status = status, cred_id = cred.id, model = model_name })
    return "cred", error_of(text, "insufficient_quota", 429, retry_headers(QUOTA_COOLDOWN_SECS))
  end
  if (status == 400 or status == 403) and is_model_access(code, message) then
    local text = who .. " has no access to model " .. tostring(model_name) .. ": " .. message
    set_cooldown(cred.id, model_name, ACCESS_COOLDOWN_SECS, text, "permission_denied", 403)
    llm_router.log("warn", "model access denied", { status = status, cred_id = cred.id, model = model_name })
    return "cred", error_of(text, "permission_denied", 403)
  end
  if status == 403 then
    llm_router.log("debug", "access denied", { status = 403, cred_id = cred.id, code = code })
    return "proxy", error_of("Alibaba Model Studio refused the request: " .. message, "server_error", 502)
  end
  if status == 429 then
    local headers = resp.headers or {}
    local gone = code == "insufficient_quota"
    local wait = gone and QUOTA_COOLDOWN_SECS or tonumber(headers["retry-after"]) or DEFAULT_COOLDOWN_SECS
    local text = gone and (who .. " is out of quota for " .. tostring(model_name) .. ": " .. message) or message
    set_cooldown(cred.id, model_name, wait, text, gone and "insufficient_quota" or "rate_limit", 429)
    llm_router.log("warn", gone and "alibaba quota exhausted" or "alibaba rate limited",
      { status = 429, cred_id = cred.id, model = model_name, wait = wait })
    return "cred", error_of(text, gone and "insufficient_quota" or "rate_limit", 429, retry_headers(wait))
  end
  if status == 404 then
    return "stop", error_of(message, "not_found", 404)
  end
  if status == 400 or status == 413 or status == 422 then
    llm_router.log("error", "request rejected", { status = status, code = code, cred_id = cred.id })
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

-- Credential x exit loop shared by every generation path.
local function attempt(ctx, model_name, send)
  local usable, total, min_wait, entry = plan_credentials(model_name)
  if total == 0 then
    return nil, error_of("no Alibaba Model Studio keys configured", "server_error", 503)
  end
  if #usable == 0 then
    local text = entry and entry.reason or ("every Alibaba Model Studio key is rate limited on " .. tostring(model_name))
    return nil, error_of(text, entry and entry.code or "rate_limit", entry and entry.status or 429, retry_headers(min_wait))
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
  local exhausted = last_err or error_of("all Alibaba Model Studio keys exhausted", "server_error", 502)
  llm_router.log("error", "all credentials exhausted", { reason = tostring(exhausted.message or "request failed"), code = exhausted.code })
  return nil, exhausted
end

-- ---------------------------------------------------------------------------
-- Reasoning
-- ---------------------------------------------------------------------------

-- Qwen thinking is not one knob. Checked live 2026-11-12:
--   effort models (qwen3.8-*): reasoning_effort none|minimal|low|medium|high|xhigh|max
--     is honoured (none turns thinking off); it cannot be combined with
--     thinking_budget. The router scale maps one to one.
--   switch models (qwen3.5-3.7, qwen-plus/flash, qwen3-vl-plus, deepseek-v4,
--     glm-5.1/5.2, kimi): enable_thinking true/false, plus thinking_budget
--     (a real cap on thinking tokens) everywhere except Kimi. reasoning_effort
--     is silently ignored there, so only `max` (= on, full thinking) and
--     `none` (= off) are advertised; qwen-plus and qwen-flash are off unless asked.
--   forced models (glm-5.3*, kimi-k2.7-code, qwen3.8-2.4t-a95b,
--     qwen3.7-max-preview, qwen3.7-max-2026-05-17): thinking cannot be
--     turned off (`enable_thinking` is "restricted to True").
-- `max_tokens` bounds the answer only; thinking tokens come on top, so a
-- reasoning.max_tokens budget is the only cap on them.
local EFFORT_ORDER = { "max", "high", "medium", "low", "none" }

local FORCED = {
  ["glm-5.3"] = true, ["glm-5.3-prime"] = true, ["kimi-k2.7-code"] = true,
  ["qwen3.8-2.4t-a95b"] = true, ["qwen3.7-max-preview"] = true, ["qwen3.7-max-2026-05-17"] = true,
}

local function bare_name(model_name)
  local lower = tostring(model_name):lower()
  return (lower:gsub("^.*/", ""))
end

-- { effort, can_off, budget, default_on } or nil for a non-reasoning model.
-- `has_cap`: what the model list says about reasoning: true/false for a card,
-- nil at request time (no list at hand). qwen-plus, qwen-flash and
-- qwen3-vl-plus have old non-reasoning snapshots, so the list decides there;
-- at request time they count as reasoning-capable. Names the rules do not
-- know count as reasoning only when the list says so.
local function reasoning_spec(model_name, has_cap)
  local n = bare_name(model_name)
  if n:find("character", 1, true) then return nil end
  local forced = FORCED[n] == true
  local effort = n:find("^qwen3%.8%-") ~= nil
  -- The 3.5 omni models do not document thinking control.
  if n:find("omni", 1, true) and not effort and not has_cap then return nil end
  local known = effort or forced
    or n:find("^qwen3%.[5-9]") ~= nil or n:find("^deepseek%-v4") ~= nil or n:find("^glm%-5%.") ~= nil
    or n:find("^kimi%-k") ~= nil
  local by_list = n:find("^qwen%-plus") ~= nil or n:find("^qwen%-flash") ~= nil or n:find("^qwen3%-vl%-plus") ~= nil
  if by_list and has_cap ~= false then known = true end
  if not known and not has_cap then return nil end
  local default_on = forced or n:find("^qwen3%.[5-9]") ~= nil or n:find("^deepseek%-v4") ~= nil
    or n:find("^glm%-5%.") ~= nil or n:find("^kimi%-k") ~= nil
  return {
    effort = effort,
    can_off = not forced,
    budget = known and n:find("^kimi") == nil,
    default_on = default_on,
  }
end

-- router level -> true (accepted). Effort models take all five; switch
-- models only on and off: on is the model's full thinking, so `max`.
local function effort_levels(spec)
  local levels = {}
  if spec.effort then
    levels.max, levels.high, levels.medium, levels.low = true, true, true, true
  else
    levels.max = true
  end
  if spec.can_off then levels.none = true end
  return levels
end

local function effort_list(levels)
  local list = {}
  for _, effort in ipairs(EFFORT_ORDER) do
    if levels[effort] then list[#list + 1] = effort end
  end
  return list
end

local function reasoning_info(spec)
  local info = {
    supported_efforts = effort_list(effort_levels(spec)),
    default_enabled = spec.default_on,
    mandatory = not spec.can_off,
    supports_max_tokens = spec.budget,
  }
  if not spec.effort then info.default_effort = spec.default_on and "max" or "none" end
  return info
end

-- Upstream reasoning fields for the request: a table (maybe empty) or nil, error.
local function reasoning_params(request)
  local model_name = request.model_name or ""
  local effort = request.reasoning_effort
  local r = type(request.reasoning) == "table" and request.reasoning or {}
  if effort == nil and type(r.effort) == "string" then effort = r.effort end
  if effort == nil and r.enabled == false then effort = "none" end
  if type(effort) ~= "string" or effort == "" or effort == "default" then effort = nil end
  local budget = type(r.max_tokens) == "number" and r.max_tokens or nil
  local spec = reasoning_spec(model_name, false)
  local out = {}

  if not spec then
    -- Not a reasoning model: "off" is trivially true, anything else is not.
    if effort ~= nil and effort ~= "none" then
      return nil, invalid_request("reasoning_effort " .. effort .. " is not supported by model " .. model_name .. "; supported: none")
    end
    if budget then
      return nil, invalid_request("model " .. model_name .. " takes no reasoning.max_tokens")
    end
    return out, nil
  end

  local levels = effort_levels(spec)
  if effort ~= nil and not levels[effort] then
    return nil, invalid_request("reasoning_effort " .. effort .. " is not supported by model " .. model_name
      .. "; supported: " .. table.concat(effort_list(levels), ", "))
  end
  if budget then
    if not spec.budget then
      return nil, invalid_request("model " .. model_name .. " takes no reasoning.max_tokens; use reasoning_effort")
    end
    if effort == "none" then
      return nil, invalid_request("reasoning.max_tokens needs reasoning on, but reasoning_effort is none")
    end
    if spec.effort and effort ~= nil then
      return nil, invalid_request("model " .. model_name .. " takes either reasoning_effort or reasoning.max_tokens, not both")
    end
    out.thinking_budget = math.max(1, math.floor(budget))
  end
  if effort ~= nil then
    if spec.effort then
      out.reasoning_effort = effort
    else
      out.enable_thinking = effort ~= "none"
    end
  elseif budget and not spec.effort then
    -- A budget only means something with thinking on (qwen-plus is off by default).
    out.enable_thinking = true
  end
  return out, nil
end

-- reasoning.exclude: the model still reasons; its text is dropped here.
local function reasoning_excluded(request)
  return type(request.reasoning) == "table" and request.reasoning.exclude == true
end

-- ---------------------------------------------------------------------------
-- Payload
-- ---------------------------------------------------------------------------

-- Fields the upstream documents for the OpenAI-compatible endpoint. It
-- ignores strangers silently (verbosity, metadata), but a whitelist keeps the
-- contract explicit. `max_completion_tokens` is NOT honoured upstream
-- (checked: it did not cap the answer), so the limit always goes out as
-- `max_tokens`.
local PASS_FIELDS = {
  "tools", "tool_choice", "parallel_tool_calls", "temperature", "top_p", "n", "stop", "seed",
  "presence_penalty", "logprobs", "top_logprobs", "response_format",
}
local MESSAGE_FIELDS = { "role", "content", "name", "tool_calls", "tool_call_id" }

local function build_messages(messages)
  local out = {}
  for i, m in ipairs(messages or {}) do
    local msg = {}
    for _, k in ipairs(MESSAGE_FIELDS) do msg[k] = m[k] end
    -- Earlier chain-of-thought goes back as `reasoning_content` (accepted;
    -- only read when preserve_thinking is on, which is not requested).
    -- `refusal` is dropped.
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
  local extra, err = reasoning_params(request)
  if err then return nil, err end
  local payload = { model = request.model_name, messages = build_messages(request.messages), stream = stream }
  for _, k in ipairs(PASS_FIELDS) do payload[k] = request[k] end
  payload.max_tokens = request.max_tokens or request.max_completion_tokens
  if payload.seed == nil then payload.seed = request.random_seed end
  if stream then
    -- Usage arrives in a last chunk with empty choices, only on request.
    local opts = type(request.stream_options) == "table" and request.stream_options or {}
    payload.stream_options = { include_usage = opts.include_usage ~= false }
  end
  for k, v in pairs(extra) do payload[k] = v end
  return payload, nil
end

-- ---------------------------------------------------------------------------
-- Model catalog
-- ---------------------------------------------------------------------------

-- The keyed /compatible-mode/v1/models says which models the key lists (ids
-- only); DashScope's /api/v1/models describes them (context, output cap,
-- modalities, features, capabilities). A listed id without a description is
-- skipped: nothing says what it is.
local function catalog_by_id(ctx, client, cred)
  local by_id = {}
  for page = 1, 6 do
    local resp, err = pooled_request(ctx, client, {
      method = "GET",
      url = host_of(cred) .. "/api/v1/models?page_size=100&page_no=" .. tostring(page),
      headers = bearer(api_key_of(cred.data)),
    })
    if err or resp.status ~= 200 then
      llm_router.log("warn", "model catalog unavailable", { status = resp and resp.status or nil })
      return by_id
    end
    local ok, body = pcall(json.decode, resp.body or "")
    local models = ok and type(body) == "table" and type(body.output) == "table" and body.output.models or nil
    if type(models) ~= "table" or #models == 0 then break end
    for _, m in ipairs(models) do
      if type(m) == "table" and type(m.model) == "string" then by_id[m.model] = m end
    end
    if #models < 100 then break end
  end
  return by_id
end

local function set_of(list)
  local set = {}
  if type(list) == "table" then
    for _, v in ipairs(list) do
      if type(v) == "string" then set[v] = true end
    end
  end
  return set
end

local function trim_text(text, limit)
  if type(text) ~= "string" or text == "" then return nil end
  if #text <= limit then return text end
  return text:sub(1, limit - 3) .. "..."
end

-- A card for a chat model, an embedding model, or nil for everything else
-- (images, speech, realtime, translation...).
local function model_info_of(id, meta)
  local caps = set_of(meta.capabilities)
  local features = set_of(meta.features)
  local info = type(meta.model_info) == "table" and meta.model_info or {}
  local modal = type(meta.inference_metadata) == "table" and meta.inference_metadata or {}
  local card = {
    name = id,
    display_name = (type(meta.name) == "string" and meta.name ~= "") and meta.name or id,
    description = trim_text(meta.description, 400),
    context_window = tonumber(info.context_window) or 0,
    max_tokens = tonumber(info.max_output_tokens) or 0,
  }
  if caps.TR then
    card.input_modalities = { "text" }
    card.output_modalities = { "embedding" }
    card.supported_parameters = { "dimensions" }
    card.endpoints = { "embeddings" }
    return card
  end
  if not (caps.TG or caps.VU or caps["Multimodal-Omni"]) then return nil end
  local params = { "temperature", "top_p", "max_tokens", "stop", "seed", "presence_penalty" }
  if features["function-calling"] then
    params[#params + 1] = "tools"
    params[#params + 1] = "tool_choice"
    params[#params + 1] = "parallel_tool_calls"
  end
  if features["structured-outputs"] then params[#params + 1] = "response_format" end
  local spec = reasoning_spec(id, caps.Reasoning == true)
  if spec then
    params[#params + 1] = "reasoning"
    params[#params + 1] = "reasoning_effort"
    card.reasoning = reasoning_info(spec)
  end
  -- Image parts are forwarded as they are; audio and video input of the omni
  -- models is not wired up here, so only "image" is claimed.
  local input = { "text" }
  if set_of(modal.request_modality).Image then input[#input + 1] = "image" end
  card.input_modalities = input
  card.output_modalities = { "text" }
  card.supported_parameters = params
  card.endpoints = { "chat/completions" }
  return card
end

-- ---------------------------------------------------------------------------
-- Plugin
-- ---------------------------------------------------------------------------

local function embed_payload(request)
  local payload = { model = request.model_name, input = request.input }
  payload.dimensions = request.dimensions
  payload.encoding_format = request.encoding_format
  return payload
end

llm_router.register("alibaba", {
  icon = "https://www.alibabacloud.com/favicon.ico",

  proxy_schema = {},

  locales = {
    ["en"] = {
      ["alibaba.subtitle"] = "Create an API key in Model Studio. Keys, models and quotas belong to one region.",
      ["alibaba.open_console"] = "Open Model Studio",
      ["alibaba.region"] = "Region",
      ["alibaba.region.singapore"] = "Singapore (international)",
      ["alibaba.region.beijing"] = "China (Beijing)",
      ["misc.api_key"] = "API Key",
      ["misc.save"] = "Save",
    },
    ["ru"] = {
      ["alibaba.subtitle"] = "Создайте API-ключ в Model Studio. Ключи, модели и квоты привязаны к одному региону.",
      ["alibaba.open_console"] = "Открыть Model Studio",
      ["alibaba.region"] = "Регион",
      ["alibaba.region.singapore"] = "Сингапур (международный)",
      ["alibaba.region.beijing"] = "Китай (Пекин)",
      ["misc.api_key"] = "API-ключ",
      ["misc.save"] = "Сохранить",
    },
  },

  credential_schema = {
    {
      type = "section",
      title = "Alibaba Model Studio",
      subtitle = "alibaba.subtitle",
      content = {
        { type = "link", text = "alibaba.open_console", url = "https://modelstudio.console.alibabacloud.com/?tab=dashboard#/api-key" },
        {
          type = "select", name = "region", label = "alibaba.region", value = DEFAULT_REGION,
          options = { "singapore", "beijing" },
          option_labels = { singapore = "alibaba.region.singapore", beijing = "alibaba.region.beijing" },
        },
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
    local region = data.region
    if region ~= nil and region ~= "" and REGIONS[region] == nil then
      return false, { message = "region must be one of: singapore, beijing", code = "invalid_request_error" }
    end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, { message = "no Alibaba Model Studio keys configured", code = "invalid_request_error", status = 400 }
    end
    -- Every key until one answers; the error names each key that failed and why.
    local client = llm_router.http_client({ timeout_ms = 20000 })
    local ids, winner, failures, last_status = nil, nil, {}, 502
    for _, cred in ipairs(creds) do
      local resp, err = pooled_request(ctx, client, { method = "GET", url = base_of(cred) .. "/models", headers = bearer(api_key_of(cred.data)) })
      local reason = nil
      if err then
        reason = "request failed: " .. ((type(err) == "table" and err.message) or tostring(err))
      elseif resp.status == 200 then
        local ok, page = pcall(json.decode, resp.body or "")
        if ok and type(page) == "table" and type(page.data) == "table" then
          ids, winner = {}, cred
          for _, m in ipairs(page.data) do
            if type(m) == "table" and type(m.id) == "string" then ids[#ids + 1] = m.id end
          end
          break
        end
        reason = "Alibaba Model Studio returned an unreadable model list"
      elseif resp.status == 401 then
        reason = "Alibaba Model Studio rejected the key: " .. upstream_message(resp, "invalid API key")
        last_status = 401
      else
        reason = "Alibaba Model Studio answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details")
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
    local meta = catalog_by_id(ctx, client, winner)
    local infos = {}
    for _, id in ipairs(ids) do
      -- "vendor/model" ids (kimi/kimi-k3, ZHIPU/GLM-5.3) are third-party
      -- hosted duplicates; they answered "product is not activated" for the
      -- tested key, while the plain ids work.
      if not id:find("/", 1, true) and meta[id] then
        local card = model_info_of(id, meta[id])
        if card then infos[#infos + 1] = card end
      end
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
    local resp, err = pooled_request(ctx, client, { method = "GET", url = base_of(credential) .. "/models", headers = bearer(key) })
    if err then
      return { status = "unknown", message = (type(err) == "table" and err.message) or "request failed" }
    end
    if resp.status == 200 then return { status = "healthy" } end
    if resp.status == 401 then
      return { status = "unhealthy", message = "Alibaba Model Studio rejected the key: " .. upstream_message(resp, "invalid API key") }
    end
    return { status = "unknown", message = "Alibaba Model Studio answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details") }
  end,

  -- Model existence with zero generation spend: GET /models/{id} per key.
  -- It says the model exists, not that the key may use it (an unactivated
  -- model still lists). Healthy wins at once; unhealthy needs every tried
  -- key to miss the model.
  check_model = function(ctx, model_ref)
    local endpoint = model_ref.endpoint or "chat"
    if endpoint ~= "chat" and endpoint ~= "vision" and endpoint ~= "embeddings" then
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
        local resp, err = pooled_request(ctx, client, { method = "GET", url = base_of(cred) .. "/models/" .. model_name, headers = bearer(key) })
        if err then
          seen_unknown = seen_unknown or ((type(err) == "table" and err.message) or "request failed")
        elseif resp.status == 200 then
          return { status = "healthy", response = "ok" }
        elseif resp.status == 404 then
          missing = missing + 1
        elseif resp.status == 429 or resp.status == 402 then
          seen_quota = true
        else
          seen_unknown = seen_unknown or ("Alibaba Model Studio answered " .. tostring(resp.status) .. ": " .. upstream_message(resp, "no details"))
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
        url = base_of(cred) .. "/chat/completions",
        headers = bearer(key, "application/json"),
        body = body,
        proxy_url = proxy_url,
      })
      return classify(ctx, cred, request.model_name, r, e)
    end)
    if not resp then return nil, err end
    local ok, out = pcall(json.decode, resp.body or "")
    if not ok or type(out) ~= "table" then
      return nil, error_of("Alibaba Model Studio returned invalid JSON", "server_error", 502)
    end
    if reasoning_excluded(request) then
      for _, choice in ipairs(out.choices or {}) do
        if type(choice.message) == "table" then
          choice.message.reasoning_content = nil
          choice.message.reasoning = nil
        end
      end
    end
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
        url = base_of(cred) .. "/chat/completions",
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

  embed = function(ctx, request)
    local body = json.encode(embed_payload(request))
    local client = llm_router.http_client({ timeout_ms = 120000 })
    local resp, err = attempt(ctx, request.model_name, function(cred, key, proxy_url)
      local r, e = client:request({
        method = "POST",
        url = base_of(cred) .. "/embeddings",
        headers = bearer(key, "application/json"),
        body = body,
        proxy_url = proxy_url,
      })
      return classify(ctx, cred, request.model_name, r, e)
    end)
    if not resp then return nil, err end
    local ok, out = pcall(json.decode, resp.body or "")
    if not ok or type(out) ~= "table" or type(out.data) ~= "table" then
      return nil, error_of("Alibaba Model Studio returned invalid JSON", "server_error", 502)
    end
    out.model = request.model
    return out
  end,
})
