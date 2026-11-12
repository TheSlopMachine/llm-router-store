--- @plugin Google AI Studio
--- @author TheSlopMachine
--- @version 6.1.0
--- @plugin_api 1.6
--- @allow_host generativelanguage.googleapis.com
--- @description Google Gemini models via the AI Studio API: chat, streaming, tools, embeddings, speech, transcription and image generation

-- Provider configuration read from ctx.provider_config (all optional):
--   proxy.pool                    dashboard proxy switch; absent means direct
--   disable_failed_credentials    dashboard switch; rejected keys are disabled instead of parked
--   google_tools                  built-in tools to enable on every request, e.g. {"googleSearch"}
--   mix_builtin_tools             Gemini 3+ only: allow built-in tools next to function tools
--   signature_in_tool_id          default true; false stops thought signatures riding in tool_call ids
--
-- Layout: helpers, schema/payload translation, response translation, the
-- credential/exit failover runner, then one handler per endpoint.

local BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

local CHAT_TIMEOUT_MS = 300000 -- grounded or thinking generations outlive the 60s router default
local STREAM_TIMEOUT_MS = 900000 -- timeout_ms bounds the whole stream, so streams get the maximum
local MEDIA_TIMEOUT_MS = 180000
local EMBED_TIMEOUT_MS = 60000
local PROBE_TIMEOUT_MS = 45000
local META_TIMEOUT_MS = 20000

local EMBED_BATCH_LIMIT = 100 -- batchEmbedContents accepts at most 100 requests per call
local MAX_INLINE_AUDIO_BYTES = 14 * 1024 * 1024 -- inline request payloads cap near 20 MB after base64
local MAX_IMAGES_PER_CALL = 10

local KEY_BENCH_SECS = 300
local DEFAULT_COOLDOWN_SECS = 60
local DAILY_COOLDOWN_SECS = 900
local ZERO_QUOTA_COOLDOWN_SECS = 3600
local MAX_COOLDOWN_SECS = 86400
local DEAD_TTL_SECS = 86400

local COOLDOWN_SCOPE = "rate_limit"
local FREE_TIER_REASON = "free tier"
local DEAD_SCOPE = "model_dead"

-- Gemini is served in most countries (not RU, BY, CN, IR and a few others): any
-- exit from a country where free proxies are common is acceptable.
local PROXY_COUNTRIES =
  { "US", "DE", "FR", "GB", "NL", "CA", "SG", "JP", "IN", "BR", "PL", "ID", "TR", "SE", "IT", "ES", "KR", "AU" }

-- ---------------------------------------------------------------------------
-- Small helpers
-- ---------------------------------------------------------------------------

local function trunc(value, limit)
  local s = tostring(value == nil and "" or value)
  if #s > limit then
    return s:sub(1, limit)
  end
  return s
end

local function api_key_of(data)
  if type(data) ~= "table" or type(data.api_key) ~= "string" then
    return ""
  end
  return (data.api_key:match("^%s*(.-)%s*$"))
end

local function error_of(message, code, status, headers)
  local e = { message = message, code = code or "server_error", status = status or 502 }
  if headers ~= nil then
    e.headers = headers
  end
  return e
end

local function invalid_request(message)
  return error_of(message, "invalid_request_error", 400)
end

local function safe_decode(body)
  if type(body) ~= "string" or body == "" then
    return nil
  end
  local ok, value = pcall(json.decode, body)
  if ok and type(value) == "table" then
    return value
  end
  return nil
end

local function has_any(haystack, needles)
  for _, needle in ipairs(needles) do
    if haystack:find(needle, 1, true) ~= nil then
      return true
    end
  end
  return false
end

local function clamp(n, low, high)
  return math.max(low, math.min(high, n))
end

local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- Proxy URLs may embed user:pass@; never persist those.
local function redact_url(url)
  if type(url) ~= "string" or url == "" then
    return "direct"
  end
  return (url:gsub("^(%a[%w+.-]*://)[^/@]*@", "%1***@"))
end

local function url_encode(s)
  return (tostring(s):gsub("[^%w%-_%.~]", function(c)
    return string.format("%%%02X", c:byte())
  end))
end

local function random_id()
  local ok, hex = pcall(llm_router.random_hex, 6)
  if ok and type(hex) == "string" and hex ~= "" then
    return hex
  end
  return tostring(os.time())
end

local function bare_model(name)
  return (tostring(name or ""):gsub("^models/", ""))
end

local function generate_url(model_name)
  return BASE_URL .. "/models/" .. model_name .. ":generateContent"
end

-- ---------------------------------------------------------------------------
-- Model families
-- ---------------------------------------------------------------------------

-- Returns major, minor for "gemini-<major>[.<minor>]-..." names, else nil.
local function model_generation(name)
  local major, minor = tostring(name):lower():match("^gemini%-(%d+)%.?(%d*)")
  if not major then
    return nil
  end
  return tonumber(major), tonumber(minor) or 0
end

local function is_gemini3(name)
  local major = model_generation(name)
  return major ~= nil and major >= 3
end

local function is_gemma(name)
  return tostring(name):lower():match("^gemma") ~= nil
end
local function is_tts_model(name)
  return tostring(name):lower():find("tts", 1, true) ~= nil
end
local function is_embedding_model(name)
  return tostring(name):lower():find("embedding", 1, true) ~= nil
end
local function is_transcribe_model(name)
  return tostring(name):lower():find("transcribe", 1, true) ~= nil
end

-- Nano Banana family: chats and renders images.
local function is_image_model(name)
  local lower = tostring(name):lower()
  return lower:find("image", 1, true) ~= nil or lower:find("nano-banana", 1, true) ~= nil
end

-- "level": Gemini 3+ thinkingLevel. "budget": Gemini 2.5 thinkingBudget.
-- nil: the family takes no thinking configuration (sending one is a 400).
local function thinking_mode(name)
  local lower = tostring(name):lower()
  if is_gemma(lower) then
    -- Gemma 4 toggles thinking through thinkingLevel; earlier Gemma has none.
    return (tonumber(lower:match("^gemma%-(%d+)") or "0") or 0) >= 4 and "level" or nil
  end
  if is_tts_model(lower) or is_embedding_model(lower) or is_transcribe_model(lower) then
    return nil
  end
  if lower:find("image", 1, true) then
    return nil
  end
  local major, minor = model_generation(lower)
  if major == nil then
    return nil
  end
  if major >= 3 then
    return "level"
  end
  if major == 2 and minor >= 5 then
    return "budget"
  end
  return nil
end

-- ---------------------------------------------------------------------------
-- Credential state: per key+model cooldowns and dead-model memory
-- ---------------------------------------------------------------------------

-- Gemini quotas bind to the key+model pair, so a 429 must never park the
-- credential (parked rows vanish from credentials.list() for every model).
-- Cooldowns live in plugin storage with a TTL; only requests for the same
-- model skip a cooling credential.
local function pair_key(cred_id, model_name)
  return tostring(cred_id) .. ":" .. bare_model(model_name)
end

local function set_cooldown(cred_id, model_name, secs, reason)
  secs = clamp(math.ceil(tonumber(secs) or DEFAULT_COOLDOWN_SECS), 1, MAX_COOLDOWN_SECS)
  local now = os.time()
  pcall(
    llm_router.storage.set,
    COOLDOWN_SCOPE,
    pair_key(cred_id, model_name),
    { at = now, ["until"] = now + secs, reason = reason or "rate limited" },
    { ttl = secs }
  )
end

-- Seconds left on a cooldown plus its reason, or nil when the pair is free.
local function cooldown_remaining(cred_id, model_name)
  local ok, value = pcall(llm_router.storage.get, COOLDOWN_SCOPE, pair_key(cred_id, model_name))
  if not ok or type(value) ~= "table" then
    return nil
  end
  if value["until"] == nil then
    return 1, value.reason
  end -- row from before expiry stamps; the TTL still bounds it
  local remaining = (tonumber(value["until"]) or 0) - os.time()
  if remaining > 0 then
    return remaining, value.reason
  end
  return nil
end

-- Google stops serving some models to a project ("no longer available to new
-- users"). That is per key, so it is remembered per credential+model for a day.
local function mark_model_dead(cred_id, model_name)
  pcall(
    llm_router.storage.set,
    DEAD_SCOPE,
    pair_key(cred_id, model_name),
    { at = os.time(), reason = "no longer available" },
    { ttl = DEAD_TTL_SECS }
  )
end

local function model_dead_for(cred_id, model_name)
  local ok, value = pcall(llm_router.storage.get, DEAD_SCOPE, pair_key(cred_id, model_name))
  return ok and value ~= nil
end

local function dead_everywhere(creds, model_name)
  if #creds == 0 then
    return false
  end
  for _, cred in ipairs(creds) do
    if not model_dead_for(cred.id, model_name) then
      return false
    end
  end
  return true
end

-- One credentials.list() pass: (usable, total, min_wait, all_dead, all_free_tier).
-- total counts keys not dead for this model; an empty usable set with a
-- positive total means every key is cooling down on this model's quota, never
-- a credential-wide outage. all_free_tier: every cooling key is cooling
-- because its free tier does not offer the model.
local function plan_credentials(model_name)
  local all = llm_router.credentials.list()
  local usable, total, dead, min_wait, free_tier = {}, 0, 0, nil, 0
  for _, cred in ipairs(all) do
    if model_dead_for(cred.id, model_name) then
      dead = dead + 1
    else
      total = total + 1
      local remaining, reason = cooldown_remaining(cred.id, model_name)
      if remaining ~= nil then
        min_wait = min_wait and math.min(min_wait, remaining) or remaining
        if reason == FREE_TIER_REASON then
          free_tier = free_tier + 1
        end
      else
        usable[#usable + 1] = cred
      end
    end
  end
  return usable, total, min_wait, (#all > 0 and dead == #all), (free_tier > 0 and free_tier == total)
end

-- Dead-key bench: shared disable when automation is on (visible in the
-- dashboard, global to every path), unified cooldown park otherwise.
local function bench(ctx, cred_id, reason, wait_secs)
  local cfg = ctx.provider_config
  if type(cfg) == "table" and cfg.disable_failed_credentials == true then
    if pcall(llm_router.credentials.disable, cred_id, reason) then
      return
    end
  end
  pcall(llm_router.credentials.park, cred_id, wait_secs, reason)
end

-- ---------------------------------------------------------------------------
-- Errors
-- ---------------------------------------------------------------------------

local function retry_headers(secs)
  if type(secs) ~= "number" or secs < 1 then
    return nil
  end
  return { ["Retry-After"] = tostring(math.ceil(secs)) }
end

-- 429 naming a free-tier quota of 0: the key's tier does not offer the model
-- at all (a paid project gets it with the same key, so this is not final).
local function is_free_tier_block(status, body)
  return status == 429 and body:find("free_tier", 1, true) ~= nil and body:find("limit: 0", 1, true) ~= nil
end

-- Agent models (Antigravity, Deep Research) answer generateContent with this.
local function is_interactions_only(status, lower)
  return status == 400 and lower:find("only supports interactions api", 1, true) ~= nil
end

local function free_tier_error()
  return error_of("gemini model is not available on the free tier of this key", "model_unavailable", 403)
end

local function interactions_only_error()
  return error_of(
    "gemini model works only through the Interactions API, which this plugin does not use",
    "model_unavailable",
    400
  )
end

local function rate_limited_error(wait)
  return error_of("gemini rate limited", "rate_limit", 429, retry_headers(wait))
end

-- code=not_found makes the router drop the model from its cache.
local function model_gone_error()
  return error_of("gemini model is no longer available to these keys", "not_found", 404)
end

local function content_filter_error(reason)
  return error_of("prompt blocked: " .. tostring(reason), "content_filter", 400)
end

-- Upstream message, trimmed, for client-facing 4xx detail.
local function upstream_message(body)
  local g = safe_decode(body)
  if g and type(g.error) == "table" and type(g.error.message) == "string" then
    return trunc(g.error.message, 300)
  end
  return nil
end

local KEY_REJECT_PATTERNS = {
  "api key not valid",
  "api_key_invalid",
  "invalid api key",
  "api key expired",
  "api key not found",
  "unauthenticated",
}
local KEY_FORBIDDEN_PATTERNS = {
  "api key",
  "api_key",
  "leaked",
  "suspended",
  "denied access",
  "service_disabled",
  "has not been used",
  "is disabled",
}
local GEO_PATTERNS = { "location is not supported", "not available in your country", "not supported in your region" }

local function is_key_rejection(status, lower)
  if status == 401 then
    return true
  end
  if status == 400 then
    return has_any(lower, KEY_REJECT_PATTERNS)
  end
  if status == 403 then
    return has_any(lower, KEY_FORBIDDEN_PATTERNS)
  end
  return false
end

local function is_geo_block(status, lower)
  return status == 400 and has_any(lower, GEO_PATTERNS)
end

-- Wait before retrying a 429: Retry-After header, then Google's RetryInfo
-- detail, then the "retry in 12.3s" sentence. Daily and zero quotas wait longer.
local function rate_limit_wait(resp)
  local body = tostring(resp.body or "")
  local wait = nil
  local headers = resp.headers or {}
  local header = tonumber(headers["retry-after"])
  if header ~= nil and header > 0 then
    wait = header
  end
  if wait == nil then
    local g = safe_decode(body)
    local details = g and type(g.error) == "table" and g.error.details or nil
    if type(details) == "table" then
      for _, d in ipairs(details) do
        if type(d) == "table" and type(d.retryDelay) == "string" then
          local seconds = tonumber(d.retryDelay:match("^([%d%.]+)s$") or "")
          if seconds ~= nil and seconds > 0 then
            wait = seconds
          end
        end
      end
    end
  end
  if wait == nil then
    local amount, unit = body:match("retry in ([%d%.]+)%s*(m?s)")
    amount = tonumber(amount)
    if amount ~= nil and amount > 0 then
      wait = (unit == "ms") and (amount / 1000) or amount
    end
  end
  wait = math.ceil(wait or DEFAULT_COOLDOWN_SECS)
  local lower = body:lower()
  if body:find('"quotaValue"%s*:%s*"0"') then
    wait = math.max(wait, ZERO_QUOTA_COOLDOWN_SECS)
  elseif lower:find("perday", 1, true) or lower:find("per day", 1, true) then
    wait = math.max(wait, DAILY_COOLDOWN_SECS)
  end
  return clamp(wait, 1, MAX_COOLDOWN_SECS)
end

-- Failure log fields. `body` is the capped upstream text.
local function failure_keys(env, status, code, cred_id, body, event)
  local keys = { operation = env.op, model = env.model, cred_id = cred_id, status = status, code = code, event = event }
  if body ~= nil and body ~= "" then
    keys.body = trunc(body, 1000)
  end
  return keys
end

-- Maps a non-200 upstream answer to a loop decision plus the client error:
--   "retry_exit" retry the same credential on the next exit
--   "next_cred"  move on to the next credential
--   "done"       return the terminal error to the client at once
-- A third value tags exit-level causes ("geo") so the runner can skip the exit.
local function map_upstream(env, resp, cred_id)
  local status = resp.status
  local body = tostring(resp.body or "")
  local lower = body:lower()

  if is_geo_block(status, lower) then
    llm_router.log(
      "warn",
      "location blocked on exit",
      failure_keys(env, status, "server_error", cred_id, body, "http_error")
    )
    return "retry_exit", error_of("gemini location block on current exit", "server_error", 502), "geo"
  end

  if is_key_rejection(status, lower) then
    bench(env.ctx, cred_id, "gemini rejected the api key", KEY_BENCH_SECS)
    llm_router.log(
      "warn",
      "api key rejected",
      failure_keys(env, status, "authentication_error", cred_id, body, "forbidden")
    )
    return "next_cred", error_of("gemini rejected the api key", "authentication_error", 401)
  end

  if status == 403 then
    -- Permission problem that is not about the key itself (model not enabled
    -- for the project, for example): back off this key+model pair only.
    set_cooldown(cred_id, env.name, KEY_BENCH_SECS)
    llm_router.log(
      "warn",
      "request forbidden for key",
      failure_keys(env, status, "authentication_error", cred_id, body, "forbidden")
    )
    return "next_cred",
      error_of(
        "gemini denied this key access to the model"
          .. (upstream_message(body) and (": " .. upstream_message(body)) or ""),
        "authentication_error",
        403
      )
  end

  if status == 429 then
    local wait = rate_limit_wait(resp)
    local free_tier = is_free_tier_block(status, body)
    set_cooldown(cred_id, env.name, wait, free_tier and FREE_TIER_REASON or nil)
    if free_tier then
      local keys = failure_keys(env, status, "model_unavailable", cred_id, body, "rate_limit")
      keys.wait = wait
      llm_router.log("warn", "model not on the free tier of key", keys)
      return "next_cred", free_tier_error()
    end
    local keys = failure_keys(env, status, "rate_limit", cred_id, body, "rate_limit")
    keys.wait = wait
    llm_router.log("warn", "rate limited", keys)
    return "next_cred", rate_limited_error(wait)
  end

  if is_interactions_only(status, lower) then
    llm_router.log(
      "error",
      "model needs the interactions api",
      failure_keys(env, status, "model_unavailable", cred_id, body, "http_error")
    )
    return "done", interactions_only_error()
  end

  if status == 404 and lower:find("no longer available", 1, true) then
    mark_model_dead(cred_id, env.name)
    llm_router.log(
      "warn",
      "model no longer available to key",
      failure_keys(env, status, "not_found", cred_id, body, "http_error")
    )
    return "next_cred", model_gone_error()
  end

  if
    (status == 503 and (lower:find("unavailable", 1, true) or lower:find("high demand", 1, true)))
    or (status == 504 and lower:find("deadline_exceeded", 1, true))
  then
    -- The model itself is overloaded: other keys hit the same backend.
    llm_router.log("error", "model overloaded", failure_keys(env, status, "server_error", cred_id, body, "http_error"))
    return "done", error_of("gemini model overloaded, try again later", "server_error", status, retry_headers(5))
  end

  if status == 404 then
    llm_router.log("error", "model not found", failure_keys(env, status, "not_found", cred_id, body, "http_error"))
    return "done", error_of("gemini model not found", "not_found", 404)
  end

  if status == 413 then
    llm_router.log(
      "error",
      "request too large",
      failure_keys(env, status, "invalid_request_error", cred_id, body, "http_error")
    )
    return "done", error_of("gemini rejected the request as too large", "invalid_request_error", 413)
  end

  if status >= 400 and status < 500 then
    llm_router.log(
      "error",
      "request rejected",
      failure_keys(env, status, "invalid_request_error", cred_id, body, "http_error")
    )
    local detail = upstream_message(body)
    return "done",
      error_of(
        "gemini rejected the request with status " .. tostring(status) .. (detail and (": " .. detail) or ""),
        "invalid_request_error",
        status
      )
  end

  llm_router.log(
    "warn",
    "unexpected upstream status",
    failure_keys(env, status, "server_error", cred_id, body, "http_error")
  )
  return "retry_exit", error_of("gemini returned status " .. tostring(status), "server_error", status)
end

-- ---------------------------------------------------------------------------
-- Proxy exits
-- ---------------------------------------------------------------------------

-- Unconfigured providers go direct: only an explicit pool selection (the
-- dashboard proxy switch) routes through pooled exits.
local function pool_of(ctx)
  local cfg = ctx.provider_config
  if
    type(cfg) == "table"
    and type(cfg.proxy) == "table"
    and type(cfg.proxy.pool) == "string"
    and cfg.proxy.pool ~= ""
  then
    return cfg.proxy.pool
  end
  return nil
end

-- Exits for one credential's attempts. A direct provider yields {{}}: one
-- attempt with no proxy_url. A configured pool never falls back to direct:
-- Gemini rejects unsupported regions, so a direct attempt would only fail
-- later with a location block.
local function pick_proxies(ctx, limit, exclude)
  local pool = pool_of(ctx)
  local options = { pool = pool, countries = PROXY_COUNTRIES, limit = limit or 3, fallback = "fail" }
  if pool ~= nil and type(exclude) == "table" and #exclude > 0 then
    options.exclude = exclude
  end
  local ok, result, code = pcall(llm_router.proxies.require, options)
  if not ok then
    llm_router.log("error", "proxy acquisition failed", { reason = trunc(result, 500), operation = "proxies_require" })
    return nil, error_of("proxy acquisition failed", "server_error", 500)
  end
  if result == nil then
    if code == "disabled" then
      return nil, error_of("router proxy pools are disabled but this provider is set to use one", "server_error", 503)
    elseif code == "timeout" then
      return nil, error_of("no live proxy exit became available in time", "server_error", 503)
    elseif code == "no_match" then
      return nil, error_of("proxy pool has no matching exit for gemini", "server_error", 503)
    elseif code == "canceled" then
      return nil, error_of("request cancelled while waiting for a proxy", "server_error", 499)
    end
    return nil, error_of("proxy acquisition failed: " .. tostring(code), "server_error", 503)
  end
  if result.direct == true or type(result.proxies) ~= "table" or #result.proxies == 0 then
    return { {} }
  end
  return result.proxies
end

-- ---------------------------------------------------------------------------
-- Schema translation (OpenAI JSON Schema -> Gemini Schema subset)
-- ---------------------------------------------------------------------------

-- Keywords copied verbatim, with their expected Lua types.
local SCHEMA_SCALARS = {
  title = "string",
  description = "string",
  pattern = "string",
  minItems = "number",
  maxItems = "number",
  minLength = "number",
  maxLength = "number",
  minimum = "number",
  maximum = "number",
  minProperties = "number",
  maxProperties = "number",
}
local STRING_FORMATS = { ["date-time"] = true, enum = true }
local NUMBER_FORMATS = { int32 = true, int64 = true, float = true, double = true }

local function non_empty_schema(s)
  if type(s) == "table" and next(s) ~= nil then
    return s
  end
  return { type = "STRING" }
end

-- Gemini accepts a documented subset of OpenAPI schema. This resolves $ref /
-- $defs inline (recursive references degrade to STRING), folds JSON Schema
-- nullability into `nullable`, turns oneOf into anyOf, merges allOf, keeps
-- only string enums and the formats Gemini knows, and drops everything else
-- (additionalProperties, $schema, strict, default, ...) because Gemini rejects
-- unknown keywords.
local function sanitize_schema(root)
  if type(root) ~= "table" then
    return nil
  end
  local defs = {}
  if type(root["$defs"]) == "table" then
    defs = root["$defs"]
  elseif type(root.definitions) == "table" then
    defs = root.definitions
  end

  local walk
  walk = function(node, depth, path)
    if type(node) ~= "table" then
      return nil
    end
    if depth > 24 then
      return { type = "STRING" }
    end

    if type(node["$ref"]) == "string" then
      local name = node["$ref"]:match("^#/%$defs/(.+)$") or node["$ref"]:match("^#/definitions/(.+)$")
      local target = name and defs[name] or nil
      if type(target) ~= "table" or path[name] then
        return { type = "STRING", description = type(node.description) == "string" and node.description or nil }
      end
      local merged = shallow_copy(target)
      for k, v in pairs(node) do
        if k ~= "$ref" then
          merged[k] = v
        end
      end
      path[name] = true
      local resolved = walk(merged, depth + 1, path)
      path[name] = nil
      return resolved
    end

    local out = {}
    local nullable = node.nullable == true
    local alternatives = node.anyOf or node.oneOf

    local t = node.type
    if type(t) == "table" then
      local kinds = {}
      for _, kind in ipairs(t) do
        if kind == "null" then
          nullable = true
        elseif type(kind) == "string" then
          kinds[#kinds + 1] = kind
        end
      end
      t = nil
      if #kinds == 1 then
        t = kinds[1]
      elseif #kinds > 1 then
        alternatives = {}
        for _, kind in ipairs(kinds) do
          alternatives[#alternatives + 1] = { type = kind }
        end
      end
    end
    if type(t) == "string" then
      if t == "null" then
        nullable = true
      else
        out.type = t:upper()
      end
    end

    if type(alternatives) == "table" and #alternatives > 0 then
      local alts = {}
      for _, sub in ipairs(alternatives) do
        if type(sub) == "table" and sub.type == "null" then
          nullable = true
        else
          local s = walk(sub, depth + 1, path)
          if s ~= nil then
            alts[#alts + 1] = non_empty_schema(s)
          end
        end
      end
      if #alts == 1 then
        for k, v in pairs(alts[1]) do
          if out[k] == nil then
            out[k] = v
          end
        end
      elseif #alts > 1 then
        out.anyOf = alts
      end
    end

    local description = type(node.description) == "string" and node.description or nil
    local enum = node.enum
    if enum == nil and node.const ~= nil then
      enum = { node.const }
    end
    if type(enum) == "table" and #enum > 0 then
      local values, all_strings = {}, true
      for _, v in ipairs(enum) do
        if type(v) ~= "string" then
          all_strings = false
        end
        values[#values + 1] = tostring(v)
      end
      if out.type == "STRING" or (out.type == nil and all_strings) then
        out.type = "STRING"
        out.enum = values
      else
        -- Gemini only types enums on strings: keep the constraint as prose.
        description = (description and (description .. " ") or "")
          .. "Allowed values: "
          .. table.concat(values, ", ")
          .. "."
      end
    end

    for key, kind in pairs(SCHEMA_SCALARS) do
      if node[key] ~= nil and type(node[key]) == kind then
        out[key] = node[key]
      end
    end
    if description ~= nil then
      out.description = description
    end
    if nullable then
      out.nullable = true
    end

    if type(node.format) == "string" then
      if out.type == "STRING" and STRING_FORMATS[node.format] then
        out.format = node.format
      elseif (out.type == "NUMBER" or out.type == "INTEGER") and NUMBER_FORMATS[node.format] then
        out.format = node.format
      end
    end

    if type(node.properties) == "table" then
      local props, count = {}, 0
      for name, sub in pairs(node.properties) do
        props[name] = non_empty_schema(walk(sub, depth + 1, path))
        count = count + 1
      end
      if count > 0 then
        out.properties = props
      end
    end
    if type(node.propertyOrdering) == "table" and #node.propertyOrdering > 0 then
      out.propertyOrdering = node.propertyOrdering
    end

    if type(node.items) == "table" then
      local item = node.items
      if #item > 0 then
        item = item[1]
      end -- tuple form: Gemini has one item schema
      out.items = non_empty_schema(walk(item, depth + 1, path))
    end
    if out.type == "ARRAY" and out.items == nil then
      out.items = { type = "STRING" }
    end
    if out.type == nil and out.properties ~= nil then
      out.type = "OBJECT"
    end

    if type(node.required) == "table" and out.properties ~= nil then
      local required = {}
      for _, name in ipairs(node.required) do
        if type(name) == "string" and out.properties[name] ~= nil then
          required[#required + 1] = name
        end
      end
      if #required > 0 then
        out.required = required
      end
    end

    if type(node.allOf) == "table" then
      for _, sub in ipairs(node.allOf) do
        local s = walk(sub, depth + 1, path)
        if type(s) == "table" then
          for k, v in pairs(s) do
            if k == "properties" then
              out.properties = out.properties or {}
              for pk, pv in pairs(v) do
                out.properties[pk] = pv
              end
            elseif k == "required" then
              out.required = out.required or {}
              for _, name in ipairs(v) do
                out.required[#out.required + 1] = name
              end
            elseif out[k] == nil then
              out[k] = v
            end
          end
        end
      end
      if out.type == nil and out.properties ~= nil then
        out.type = "OBJECT"
      end
    end
    return out
  end

  return walk(root, 0, {})
end

-- Function parameters: a tool with no properties omits `parameters` (Gemini
-- rejects an OBJECT schema without properties).
local function function_parameters(params)
  local s = sanitize_schema(params)
  if s == nil or s.properties == nil then
    return nil
  end
  s.type = "OBJECT"
  return s
end

local function response_schema(schema)
  local s = sanitize_schema(schema)
  if s == nil or next(s) == nil then
    return nil
  end
  return s
end

-- ---------------------------------------------------------------------------
-- Message translation (OpenAI messages -> Gemini contents)
-- ---------------------------------------------------------------------------

-- Plain-text view of message content. Content arrives either as a string or
-- as an array of content parts. Only text is extracted here; binary parts are
-- handled structurally by user_content.
local function text_of_content(content)
  if type(content) == "string" then
    return content
  end
  if type(content) == "number" then
    return tostring(content)
  end
  if type(content) ~= "table" then
    return ""
  end
  if type(content.text) == "string" and content.text ~= "" then
    return content.text
  end
  if type(content.content) == "string" and content.content ~= "" then
    return content.content
  end
  local parts = {}
  for _, p in ipairs(content) do
    if type(p) == "string" then
      if p ~= "" then
        parts[#parts + 1] = p
      end
    elseif type(p) == "table" then
      if type(p.text) == "string" and p.text ~= "" then
        parts[#parts + 1] = p.text
      elseif type(p.content) == "string" and p.content ~= "" then
        parts[#parts + 1] = p.content
      elseif type(p.content) == "table" then
        local inner = text_of_content(p.content)
        if inner ~= "" then
          parts[#parts + 1] = inner
        end
      end
    end
  end
  return table.concat(parts, "\n")
end

-- Inline types Gemini reads (checked 2026-11-12): image, audio, video, any
-- text/* (plain, markdown, csv, html, xml, code), PDF, JSON, XML, RTF and
-- Office/OpenDocument files.
local DOCUMENT_MIMES = {
  ["application/pdf"] = true,
  ["application/json"] = true,
  ["application/xml"] = true,
  ["application/rtf"] = true,
  ["application/msword"] = true,
  ["application/vnd.ms-excel"] = true,
  ["application/vnd.ms-powerpoint"] = true,
}
local MIME_BY_EXTENSION = {
  pdf = "application/pdf",
  txt = "text/plain",
  md = "text/markdown",
  csv = "text/csv",
  tsv = "text/tab-separated-values",
  html = "text/html",
  htm = "text/html",
  xml = "text/xml",
  json = "application/json",
  rtf = "text/rtf",
  py = "text/x-python",
  js = "text/javascript",
  ts = "text/x-typescript",
  go = "text/x-go",
  lua = "text/x-lua",
  doc = "application/msword",
  docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xls = "application/vnd.ms-excel",
  xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ppt = "application/vnd.ms-powerpoint",
  pptx = "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  odt = "application/vnd.oasis.opendocument.text",
  ods = "application/vnd.oasis.opendocument.spreadsheet",
  odp = "application/vnd.oasis.opendocument.presentation",
}

local function inline_mime_ok(mime)
  local family = mime:match("^([^/]+)/")
  return family == "image"
    or family == "audio"
    or family == "video"
    or family == "text"
    or DOCUMENT_MIMES[mime] == true
    or mime:find("^application/vnd%.openxmlformats%-officedocument%.") ~= nil
    or mime:find("^application/vnd%.oasis%.opendocument%.") ~= nil
end

-- A text whose entire trimmed content is a data: URL carries binary inline
-- (fallback for clients that cannot send typed parts). A generic or missing
-- type falls back to the file name's extension. Returns mime, base64.
local function parse_data_url(s, filename)
  if type(s) ~= "string" then
    return nil
  end
  local start = s:find("%S")
  if start == nil or s:sub(start, start + 4) ~= "data:" then
    return nil
  end
  local head, b64 = s:sub(start):match("^data:(.-);base64,(.+)$")
  if not head or not b64 then
    return nil
  end
  b64 = b64:gsub("%s+$", "")
  local mime = (head:match("^([^;,]+)") or ""):lower()
  if b64 == "" then
    return nil
  end
  if not inline_mime_ok(mime) then
    local ext = tostring(filename or ""):lower():match("%.(%w+)$")
    mime = ext and MIME_BY_EXTENSION[ext]
    if not mime then
      return nil
    end
  end
  return mime, b64
end

local AUDIO_MIME_BY_FORMAT = {
  wav = "audio/wav",
  mp3 = "audio/mp3",
  aiff = "audio/aiff",
  aac = "audio/aac",
  ogg = "audio/ogg",
  flac = "audio/flac",
}

-- Gemini 3 rejects replayed function calls that lack the signature they were
-- issued with. Histories built elsewhere carry none; this documented dummy
-- value skips validation for that turn at some cost in reasoning quality.
local DUMMY_SIGNATURE = "skip_thought_signature_validator"

-- Tool-call ids minted by this plugin look like "<id>|<thought signature>".
local function split_tool_id(id)
  if type(id) ~= "string" then
    return "", nil
  end
  local bar = id:find("|", 1, true)
  if not bar then
    return id, nil
  end
  local signature = id:sub(bar + 1)
  if signature == "" then
    signature = nil
  end
  return id:sub(1, bar - 1), signature
end

local function signature_of_call(tc)
  local _, from_id = split_tool_id(tc.id)
  if from_id then
    return from_id
  end
  local extra = tc.extra_content
  if type(extra) == "table" and type(extra.google) == "table" and type(extra.google.thought_signature) == "string" then
    return extra.google.thought_signature
  end
  if type(tc.thought_signature) == "string" and tc.thought_signature ~= "" then
    return tc.thought_signature
  end
  local specific = tc.provider_specific_fields
  if type(specific) == "table" and type(specific.thought_signature) == "string" then
    return specific.thought_signature
  end
  return nil
end

-- Returns contents, system_parts, err. Tool results are resolved against
-- assistant tool_calls seen earlier in the same history (two-pass build).
-- Reasoning summaries are not replayed: they cost input tokens and Gemini
-- carries reasoning state in thought signatures instead.
local function build_contents(messages, model_name)
  local gemini3 = is_gemini3(model_name)
  local call_names = {}
  for _, m in ipairs(messages or {}) do
    if m.role == "assistant" and type(m.tool_calls) == "table" then
      for _, tc in ipairs(m.tool_calls) do
        local fn = type(tc) == "table" and tc["function"] or nil
        if
          type(tc) == "table"
          and type(tc.id) == "string"
          and tc.id ~= ""
          and type(fn) == "table"
          and type(fn.name) == "string"
          and fn.name ~= ""
        then
          call_names[tc.id] = fn.name
          call_names[(split_tool_id(tc.id))] = fn.name
        end
      end
    end
  end

  local system_parts = {}
  local contents = {}
  local current = nil
  local current_merge = false
  local function flush()
    if current ~= nil then
      if #current.parts > 0 then
        contents[#contents + 1] = current
      end
      current = nil
      current_merge = false
    end
  end
  local function text_part(role, text)
    if text == "" then
      return
    end
    if current and current.role == role and current_merge then
      current.parts[#current.parts + 1] = { text = text }
    else
      flush()
      current = { role = role, parts = { { text = text } } }
      current_merge = true
    end
  end
  local function model_part(part)
    if not (current and current.role == "model") then
      flush()
      current = { role = "model", parts = {} }
    end
    current_merge = true
    current.parts[#current.parts + 1] = part
  end
  -- A function result always lives in its own user block; consecutive tool
  -- messages share one block.
  local function fnresp_part(fr)
    if current and current.role == "user" and not current_merge then
      current.parts[#current.parts + 1] = { functionResponse = fr }
    else
      flush()
      current = { role = "user", parts = { { functionResponse = fr } } }
      current_merge = false
    end
  end

  -- Structured walk over user content: text accumulates for merging, binary
  -- parts become inlineData in place. Returns an error string for content the
  -- plugin cannot forward (remote URLs it is not allowed to download).
  local function user_content(role, content, name)
    local texts = {}
    local named = false
    local function flush_text()
      if #texts == 0 then
        return
      end
      local text = table.concat(texts, "\n")
      texts = {}
      if not named and name and name ~= "" then
        text = "[" .. name .. "] " .. text
        named = true
      end
      text_part(role, text)
    end
    local function push_binary(part)
      flush_text()
      if not (current and current.role == role and current_merge) then
        flush()
        current = { role = role, parts = {} }
      end
      current_merge = true
      current.parts[#current.parts + 1] = part
    end
    local function add_segment(seg)
      if type(seg) == "string" then
        if seg == "" then
          return nil
        end
        local mime, b64 = parse_data_url(seg)
        if mime then
          push_binary({ inlineData = { mimeType = mime, data = b64 } })
          return nil
        end
        texts[#texts + 1] = seg
        return nil
      end
      if type(seg) ~= "table" then
        return nil
      end
      if type(seg.image_url) == "table" or type(seg.image_url) == "string" then
        local url = type(seg.image_url) == "table" and seg.image_url.url or seg.image_url
        local mime, b64 = parse_data_url(url)
        if mime then
          -- OpenAI `detail` is deliberately not mapped to mediaResolution: it is
          -- advisory and the API rejects the OpenAI level names.
          push_binary({ inlineData = { mimeType = mime, data = b64 } })
          return nil
        end
        if type(url) == "string" and url ~= "" then
          return "image URL is not a data: URL; the plugin cannot download remote images"
        end
        return nil
      end
      if type(seg.input_audio) == "table" then
        local data = seg.input_audio.data or ""
        local mime = AUDIO_MIME_BY_FORMAT[tostring(seg.input_audio.format or ""):lower()]
        if data == "" or not mime then
          return "input_audio needs base64 data and format wav, mp3, aiff, aac, ogg or flac"
        end
        push_binary({ inlineData = { mimeType = mime, data = data } })
        return nil
      end
      if type(seg.file) == "table" then
        local mime, b64 = parse_data_url(seg.file.file_data, seg.file.filename)
        if not mime then
          return "file parts need file_data as a base64 data: URL of a type gemini reads"
            .. " (image, audio, video, text, pdf, json, xml, rtf, office)"
        end
        push_binary({ inlineData = { mimeType = mime, data = b64 } })
        return nil
      end
      local text = text_of_content(seg)
      if text ~= "" then
        texts[#texts + 1] = text
      end
      return nil
    end
    if type(content) == "string" or type(content) == "number" then
      local err = add_segment(tostring(content))
      if err then
        return err
      end
    elseif type(content) == "table" then
      for _, seg in ipairs(content) do
        local err = add_segment(seg)
        if err then
          return err
        end
      end
    end
    flush_text()
    return nil
  end

  for _, m in ipairs(messages or {}) do
    local role = m.role
    if role == "system" or role == "developer" then
      local text = text_of_content(m.content)
      if m.name and m.name ~= "" and text ~= "" then
        text = "[" .. m.name .. "] " .. text
      end
      if text ~= "" then
        system_parts[#system_parts + 1] = { text = text }
      end
    elseif role == "assistant" then
      local text = text_of_content(m.content)
      local calls = type(m.tool_calls) == "table" and m.tool_calls or {}
      local refusal = type(m.refusal) == "string" and m.refusal or ""
      if text ~= "" then
        model_part({ text = text })
      end
      if text == "" and #calls == 0 and refusal ~= "" then
        model_part({ text = refusal })
      end
      local first_call = true
      for _, tc in ipairs(calls) do
        if type(tc) ~= "table" then
          return nil, nil, invalid_request("assistant tool call must be an object")
        end
        local fn = tc["function"] or {}
        if type(fn.name) ~= "string" or fn.name == "" then
          return nil, nil, invalid_request("assistant tool call is missing a function name")
        end
        local args = {}
        if type(fn.arguments) == "string" and fn.arguments ~= "" then
          local ok, parsed = pcall(json.decode, fn.arguments)
          if not ok or type(parsed) ~= "table" then
            return nil, nil, invalid_request("tool call arguments are not valid JSON for function " .. fn.name)
          end
          args = parsed
        elseif type(fn.arguments) == "table" then
          args = fn.arguments
        end
        local id, _ = split_tool_id(tc.id)
        local fc = { name = fn.name }
        if next(args) ~= nil then
          fc.args = args
        end
        if id ~= "" then
          fc.id = id
        end
        local part = { functionCall = fc }
        local signature = signature_of_call(tc)
        if signature then
          part.thoughtSignature = signature
        elseif first_call and gemini3 then
          part.thoughtSignature = DUMMY_SIGNATURE
        end
        first_call = false
        model_part(part)
      end
    elseif role == "tool" or role == "function" then
      local call_id = m.tool_call_id or ""
      local name = call_names[call_id] or call_names[(split_tool_id(call_id))]
      if not name and type(m.name) == "string" and m.name ~= "" then
        name = m.name
      end
      if not name then
        return nil, nil, invalid_request("tool result references unknown tool call id " .. tostring(call_id))
      end
      local raw = text_of_content(m.content)
      local response
      local ok, parsed = pcall(json.decode, raw)
      if raw ~= "" and ok and type(parsed) == "table" and parsed[1] == nil and next(parsed) ~= nil then
        response = parsed
      else
        response = { output = raw }
      end
      local fr = { name = name, response = response }
      local id = split_tool_id(call_id)
      if id ~= "" then
        fr.id = id
      end
      fnresp_part(fr)
    else
      local uerr = user_content("user", m.content, m.name)
      if uerr then
        return nil, nil, invalid_request(uerr)
      end
    end
  end
  flush()
  -- Gemini rejects histories ending with a model turn; nudge with a minimal
  -- user part so fall-through and plain requests validate alike.
  if #contents > 0 and contents[#contents].role == "model" then
    contents[#contents + 1] = { role = "user", parts = { { text = "Continue." } } }
  end
  if #contents == 0 then
    return nil, nil, invalid_request("messages produced no content")
  end
  return contents, system_parts, nil
end

-- ---------------------------------------------------------------------------
-- Tools
-- ---------------------------------------------------------------------------

local BUILTIN_TOOLS = { googleSearch = true, codeExecution = true, urlContext = true }
-- A request-level function tool with one of these names is the OpenAI-shaped
-- spelling of native Google Search: nothing to execute client-side.
local SEARCH_TOOL_NAMES = { web_search = true, google_search = true }
local SEARCH_TOOL_TYPES = { web_search = true, web_search_preview = true, google_search = true }

-- Returns tools, tool_config, err.
-- Before Gemini 3, built-in tools and function declarations cannot share a
-- request (400 "Multiple tools are supported only when they are all search
-- tools"). When function tools are present, a client-declared web_search stays
-- an ordinary function the client executes, and built-in tools are skipped
-- unless the operator opted into mixing on a Gemini 3+ model.
local function build_tools(request, provider_config, model_name)
  local real, search_named, request_native = {}, {}, {}
  local tools = request.tools
  if type(tools) == "table" then
    for _, t in ipairs(tools) do
      if type(t) ~= "table" then
        return nil, nil, invalid_request("tool must be an object")
      end
      local fn = t["function"]
      local name, desc, params
      if type(fn) == "table" then
        name, desc, params = fn.name, fn.description, fn.parameters
      elseif type(t.name) == "string" then
        name, desc, params = t.name, t.description, t.parameters or t.input_schema
      end
      if type(name) == "string" and name ~= "" then
        local decl = { name = name, description = (type(desc) == "string" and desc ~= "") and desc or name }
        local parameters = function_parameters(params)
        if parameters then
          decl.parameters = parameters
        end
        if SEARCH_TOOL_NAMES[name] then
          search_named[#search_named + 1] = decl
        else
          real[#real + 1] = decl
        end
      elseif type(t.type) == "string" and SEARCH_TOOL_TYPES[t.type] then
        request_native[#request_native + 1] = "googleSearch"
      else
        return nil, nil, invalid_request("tool is missing a function name")
      end
    end
  end

  local decls = real
  local native, native_set = {}, {}
  local function add_native(name)
    if not native_set[name] then
      native_set[name] = true
      native[#native + 1] = name
    end
  end
  for _, name in ipairs(request_native) do
    add_native(name)
  end
  if #real == 0 then
    if #search_named > 0 then
      add_native("googleSearch")
    end
  else
    for _, decl in ipairs(search_named) do
      decls[#decls + 1] = decl
    end
  end
  if type(provider_config) == "table" and type(provider_config.google_tools) == "table" then
    for _, name in ipairs(provider_config.google_tools) do
      if BUILTIN_TOOLS[name] then
        add_native(name)
      else
        llm_router.log(
          "warn",
          "unknown built-in tool in provider config",
          { operation = "build_tools", reason = trunc(name, 64) }
        )
      end
    end
  end
  if #decls > 0 and #native > 0 then
    local may_mix = type(provider_config) == "table"
      and provider_config.mix_builtin_tools == true
      and is_gemini3(model_name)
    if not may_mix then
      llm_router.log(
        "debug",
        "built-in tools skipped next to function tools",
        { operation = "build_tools", count = #native }
      )
      native, native_set = {}, {}
    end
  end

  local tool_list = {}
  if #decls > 0 then
    tool_list[#tool_list + 1] = { functionDeclarations = decls }
  end
  for _, name in ipairs(native) do
    tool_list[#tool_list + 1] = { [name] = {} }
  end

  local tc = request.tool_choice
  local mode, allowed = nil, nil
  local function from_kind(kind)
    if kind == "none" then
      mode = "NONE"
    elseif kind == "auto" then
      mode = "AUTO"
    elseif kind == "required" then
      mode = "ANY"
    end
  end
  if type(tc) == "string" then
    from_kind(tc)
  elseif type(tc) == "table" then
    if
      tc.type == "function"
      and type(tc["function"]) == "table"
      and type(tc["function"].name) == "string"
      and tc["function"].name ~= ""
    then
      mode = "ANY"
      allowed = { tc["function"].name }
    elseif type(tc.type) == "string" then
      from_kind(tc.type)
    end
  end
  -- ANY forces a function call; with no function declarations there is
  -- nothing to call and Gemini rejects or hangs.
  if mode == "ANY" and #decls == 0 then
    mode, allowed = nil, nil
  end
  if allowed then
    local known = false
    for _, decl in ipairs(decls) do
      if decl.name == allowed[1] then
        known = true
      end
    end
    if not known then
      allowed = nil
    end
  end
  local tool_config = nil
  if mode then
    tool_config = { functionCallingConfig = { mode = mode } }
    if allowed then
      tool_config.functionCallingConfig.allowedFunctionNames = allowed
    end
  end
  if #tool_list == 0 then
    return nil, nil, nil
  end
  return tool_list, tool_config, nil
end

-- ---------------------------------------------------------------------------
-- Generation config
-- ---------------------------------------------------------------------------

-- Router reasoning scale, highest first: max, high, medium, low, none. A model
-- accepts exactly the efforts it advertises in supported_efforts ("default"
-- means the model default and is never advertised); anything else is a 400.
-- Efforts map onto thinkingLevel per family (checked 2026-11-12):
--   Flash up to 3.6, every Flash-Lite:   high, medium, low, none=MINIMAL
--   Flash from 3.7, Pro tiers, Gemini 4: high, medium, low (thinking stays on)
--   Gemini 3 Pro:                        high, low
--   Gemma 4:                             max=HIGH (on), none=MINIMAL (off)
-- Gemini 2.5 takes only reasoning.max_tokens (thinkingBudget).
local EFFORT_ORDER = { "max", "high", "medium", "low", "none" }

local function effort_levels(name)
  local lower = tostring(name):lower()
  if is_gemma(lower) then
    return { max = "HIGH", none = "MINIMAL" }
  end
  if lower:match("^gemini%-3%-pro") then
    return { high = "HIGH", low = "LOW" }
  end
  local levels = { high = "HIGH", medium = "MEDIUM", low = "LOW" }
  if lower:find("%-pro") or lower:match("^gemini%-4") then
    return levels
  end
  local major, minor = model_generation(lower)
  if not lower:find("lite", 1, true) and major ~= nil and (major > 3 or (major == 3 and minor >= 7)) then
    return levels
  end
  levels.none = "MINIMAL"
  return levels
end

-- Advertised efforts of a level-mode model, highest first.
local function effort_list(name)
  local levels = effort_levels(name)
  local list = {}
  for _, effort in ipairs(EFFORT_ORDER) do
    if levels[effort] then
      list[#list + 1] = effort
    end
  end
  return list
end

-- OpenRouter-style reasoning.exclude: thinking still runs, its text is not returned.
local function reasoning_excluded(request)
  return type(request.reasoning) == "table" and request.reasoning.exclude == true
end

-- reasoning_effort (or reasoning.effort; reasoning.enabled=false as "none")
-- and reasoning.max_tokens, as given. Empty or "default" means the model default.
local function reasoning_request(request)
  local effort = request.reasoning_effort
  local max_tokens = nil
  local r = request.reasoning
  if type(r) == "table" then
    if effort == nil and type(r.effort) == "string" then
      effort = r.effort
    end
    if effort == nil and r.enabled == false then
      effort = "none"
    end
    if type(r.max_tokens) == "number" and r.max_tokens >= 0 then
      max_tokens = math.floor(r.max_tokens)
    end
  end
  if type(effort) ~= "string" or effort == "" or effort == "default" then
    effort = nil
  end
  return effort, max_tokens
end

-- Rejects reasoning controls the model does not take. Returns an error or nil.
local function reasoning_check(model_name, effort, max_tokens)
  local mode = thinking_mode(model_name)
  if effort ~= nil and not (mode == "level" and effort_levels(model_name)[effort]) then
    if mode ~= "level" then
      return invalid_request(
        "model "
          .. model_name
          .. " takes no reasoning_effort"
          .. (mode == "budget" and "; use reasoning.max_tokens" or "")
      )
    end
    return invalid_request(
      "reasoning_effort "
        .. effort
        .. " is not supported by model "
        .. model_name
        .. "; supported: "
        .. table.concat(effort_list(model_name), ", ")
    )
  end
  if max_tokens ~= nil and mode ~= "budget" then
    return invalid_request(
      "model " .. model_name .. " takes no reasoning.max_tokens" .. (mode == "level" and "; use reasoning_effort" or "")
    )
  end
  return nil
end

-- thinkingConfig for a request that passed reasoning_check: the effort's level
-- (Gemini 3+, Gemma 4) or the client's budget as given (Gemini 2.5); neither
-- keeps the model default. Thought summaries are requested unless the client
-- excluded them or thinking is off; Gemma returns its thoughts unasked and
-- rejects the flag. Returns nil when there is nothing to send.
local function thinking_config(model_name, effort, max_tokens, exclude)
  local mode = thinking_mode(model_name)
  if mode == nil then
    return nil
  end
  local cfg = {}
  if mode == "level" and effort ~= nil then
    cfg.thinkingLevel = effort_levels(model_name)[effort]
  elseif mode == "budget" and max_tokens ~= nil then
    cfg.thinkingBudget = max_tokens
  end
  if not exclude and not is_gemma(model_name) and cfg.thinkingLevel ~= "MINIMAL" and cfg.thinkingBudget ~= 0 then
    cfg.includeThoughts = true
  end
  if next(cfg) == nil then
    return nil
  end
  return cfg
end

local function build_generation_config(request, model_name)
  local cfg = nil
  local function ensure()
    if not cfg then
      cfg = {}
    end
    return cfg
  end
  local max_tokens = request.max_completion_tokens or request.max_tokens
  if type(max_tokens) == "number" and max_tokens > 0 then
    ensure().maxOutputTokens = math.floor(max_tokens)
  end
  if type(request.temperature) == "number" and request.temperature >= 0 then
    ensure().temperature = math.min(request.temperature, 2)
  end
  if type(request.top_p) == "number" and request.top_p >= 0 and request.top_p <= 1 then
    ensure().topP = request.top_p
  end
  if type(request.top_k) == "number" and request.top_k >= 1 then
    ensure().topK = math.floor(request.top_k)
  end
  if
    type(request.seed) == "number"
    and request.seed == math.floor(request.seed)
    and request.seed >= -2147483648
    and request.seed <= 2147483647
  then
    ensure().seed = request.seed
  end
  if request.stop ~= nil then
    local stops = {}
    if type(request.stop) == "string" then
      if request.stop ~= "" then
        stops = { request.stop }
      end
    elseif type(request.stop) == "table" then
      for _, s in ipairs(request.stop) do
        if (type(s) == "string" and s ~= "") or type(s) == "number" then
          stops[#stops + 1] = tostring(s)
        end
        if #stops >= 5 then
          break
        end
      end
    end
    if #stops > 0 then
      ensure().stopSequences = stops
    end
  end
  -- frequencyPenalty/presencePenalty/logprobs are intentionally not forwarded:
  -- Gemini rejects them on several models (400 "Penalty is not enabled") and
  -- /models exposes no per-model capability flag for them.

  local effort, budget_tokens = reasoning_request(request)
  local rerr = reasoning_check(model_name, effort, budget_tokens)
  if rerr then
    return nil, rerr
  end
  local thinking = thinking_config(model_name, effort, budget_tokens, reasoning_excluded(request))
  if thinking then
    ensure().thinkingConfig = thinking
  end

  local rf = request.response_format
  if type(rf) == "table" and type(rf.type) == "string" then
    if rf.type == "text" then
      -- omit: text/plain is the default
    elseif rf.type == "json_object" then
      ensure().responseMimeType = "application/json"
    elseif rf.type == "json_schema" then
      ensure().responseMimeType = "application/json"
      if type(rf.json_schema) == "table" then
        local schema = response_schema(rf.json_schema.schema)
        if schema then
          ensure().responseSchema = schema
        end
      end
    else
      return nil, invalid_request("unsupported response_format type " .. rf.type)
    end
  end
  return cfg, nil
end

local function build_payload(request, provider_config)
  if type(request.n) == "number" and request.n > 1 then
    return nil, invalid_request("multiple candidates (n > 1) are not supported")
  end
  local model = bare_model(request.model_name)
  local contents, system_parts, cerr = build_contents(request.messages, model)
  if cerr then
    return nil, cerr
  end
  local payload = { contents = contents }
  if #system_parts > 0 then
    payload.systemInstruction = { parts = system_parts }
  end
  local tools, tool_config, terr = build_tools(request, provider_config, model)
  if terr then
    return nil, terr
  end
  if tools then
    payload.tools = tools
  end
  if tool_config then
    payload.toolConfig = tool_config
  end
  local cfg, gerr = build_generation_config(request, model)
  if gerr then
    return nil, gerr
  end
  if cfg then
    payload.generationConfig = cfg
  end
  return payload, nil
end

-- ---------------------------------------------------------------------------
-- Response translation (Gemini candidates -> OpenAI)
-- ---------------------------------------------------------------------------

-- Finish reasons that mean policy blocked the output: OpenAI's single
-- content_filter bucket. Anything else that is not STOP or MAX_TOKENS is a
-- model fault or an unremarkable end and is not reported as a policy block.
local POLICY_FINISH = {
  SAFETY = true,
  RECITATION = true,
  BLOCKLIST = true,
  PROHIBITED_CONTENT = true,
  SPII = true,
  LANGUAGE = true,
  IMAGE_SAFETY = true,
  IMAGE_PROHIBITED_CONTENT = true,
  IMAGE_RECITATION = true,
}
local FAULT_FINISH = {
  MALFORMED_FUNCTION_CALL = true,
  UNEXPECTED_TOOL_CALL = true,
  TOO_MANY_TOOL_CALLS = true,
  MISSING_THOUGHT_SIGNATURE = true,
  MALFORMED_RESPONSE = true,
}

local function map_finish_reason(reason)
  if reason == nil or reason == "" or reason == "STOP" then
    return "stop"
  end
  if reason == "MAX_TOKENS" then
    return "length"
  end
  if POLICY_FINISH[reason] then
    return "content_filter"
  end
  return "stop"
end

-- The error for a candidate that ended in a model fault with nothing usable.
local function finish_fault(cand)
  local reason = tostring(cand.finishReason)
  local detail = type(cand.finishMessage) == "string" and (": " .. trunc(cand.finishMessage, 200)) or ""
  if reason == "MISSING_THOUGHT_SIGNATURE" then
    return invalid_request("gemini requires the thought signature of earlier function calls" .. detail)
  end
  return error_of("gemini ended the response with " .. reason .. detail, "server_error", 502)
end

-- Collect one candidate into text / reasoning / tool-call buckets shared by
-- complete and complete_stream. seq.n numbers calls across a whole stream, so
-- each call keeps a stable index and a synthesized id.
local function collect_parts(cand, out, seq, sig_in_id)
  local parts = cand and cand.content and cand.content.parts or {}
  for _, p in ipairs(parts) do
    if type(p) == "table" then
      if p.thought == true then
        if type(p.text) == "string" and p.text ~= "" then
          out.reasoning[#out.reasoning + 1] = p.text
        end
      elseif type(p.text) == "string" and p.text ~= "" then
        out.text[#out.text + 1] = p.text
      elseif type(p.functionCall) == "table" then
        local fc = p.functionCall
        local args = "{}"
        if type(fc.args) == "table" and next(fc.args) ~= nil then
          args = json.encode(fc.args)
        end
        seq.n = seq.n + 1
        local call_id = (type(fc.id) == "string" and fc.id ~= "") and fc.id
          or string.format("call_%d_%d", cand.index or 0, seq.n)
        local call = { index = seq.n - 1, id = call_id, name = fc.name or "", arguments = args }
        local signature = p.thoughtSignature or p.thought_signature
        if type(signature) == "string" and signature ~= "" then
          call.thought_signature = signature
          -- Clients that drop unknown fields still round-trip the signature.
          if sig_in_id then
            call.id = call_id .. "|" .. signature
          end
        end
        out.toolcalls[#out.toolcalls + 1] = call
      elseif type(p.inlineData) == "table" then
        -- Image/audio output has no OpenAI chat field; expose it as a data-URL
        -- markdown image so the bytes are not silently dropped.
        local mime = p.inlineData.mimeType or "application/octet-stream"
        local data = p.inlineData.data or ""
        if data ~= "" then
          out.text[#out.text + 1] = "![" .. mime .. "](data:" .. mime .. ";base64," .. data .. ")"
        end
      elseif type(p.fileData) == "table" and p.fileData.fileUri then
        out.text[#out.text + 1] = "[file](" .. tostring(p.fileData.fileUri) .. ")"
      elseif type(p.executableCode) == "table" then
        local code = p.executableCode.code or ""
        if code ~= "" then
          out.text[#out.text + 1] = "```"
            .. tostring(p.executableCode.language or ""):lower()
            .. "\n"
            .. code
            .. "\n```"
        end
      elseif type(p.codeExecutionResult) == "table" then
        local output = p.codeExecutionResult.output or ""
        if output ~= "" then
          out.text[#out.text + 1] = output
        end
      end
    end
  end
end

-- OpenAI tool_call shape for a collected call. The signature also rides in
-- extra_content, the field Google's own OpenAI-compatible layer uses.
local function openai_tool_call(tc, with_index)
  local call = { id = tc.id, type = "function", ["function"] = { name = tc.name, arguments = tc.arguments } }
  if with_index then
    call.index = tc.index
  end
  if tc.thought_signature then
    call.extra_content = { google = { thought_signature = tc.thought_signature } }
  end
  return call
end

-- Gemini counts thinking tokens apart from candidate tokens; OpenAI's
-- completion_tokens includes reasoning, and tool-use prompt tokens are prompt.
local function build_usage(meta)
  if type(meta) ~= "table" then
    return nil
  end
  local prompt = (meta.promptTokenCount or 0) + (meta.toolUsePromptTokenCount or 0)
  local thoughts = meta.thoughtsTokenCount or 0
  local completion = (meta.candidatesTokenCount or 0) + thoughts
  local usage = {
    prompt_tokens = prompt,
    completion_tokens = completion,
    total_tokens = meta.totalTokenCount or (prompt + completion),
  }
  if thoughts > 0 then
    usage.completion_tokens_details = { reasoning_tokens = thoughts }
  end
  local function modality_sum(details, want)
    local n = 0
    for _, d in ipairs(details or {}) do
      if d.modality == want and type(d.tokenCount) == "number" then
        n = n + d.tokenCount
      end
    end
    return n
  end
  local prompt_details = nil
  if meta.cachedContentTokenCount and meta.cachedContentTokenCount > 0 then
    prompt_details = { cached_tokens = meta.cachedContentTokenCount }
  end
  local prompt_audio = modality_sum(meta.promptTokensDetails, "AUDIO")
  if prompt_audio > 0 then
    prompt_details = prompt_details or {}
    prompt_details.audio_tokens = prompt_audio
  end
  if prompt_details then
    usage.prompt_tokens_details = prompt_details
  end
  local completion_audio = modality_sum(meta.candidatesTokensDetails, "AUDIO")
  if completion_audio > 0 then
    usage.completion_tokens_details = usage.completion_tokens_details or {}
    usage.completion_tokens_details.audio_tokens = completion_audio
  end
  return usage
end

local function sig_in_id_of(provider_config)
  return not (type(provider_config) == "table" and provider_config.signature_in_tool_id == false)
end

-- Unary assembly: Gemini response -> (kind, value) for the runner.
local function assemble_completion(g, full_model, sig_in_id, hide_reasoning)
  local cand = type(g.candidates) == "table" and g.candidates[1] or nil
  if cand == nil then
    if type(g.promptFeedback) == "table" and g.promptFeedback.blockReason then
      return "done", content_filter_error(g.promptFeedback.blockReason)
    end
    return "retry_exit", error_of("upstream returned no generated content", "server_error", 502)
  end
  local out = { text = {}, reasoning = {}, toolcalls = {} }
  collect_parts(cand, out, { n = 0 }, sig_in_id)
  if #out.text == 0 and #out.toolcalls == 0 and FAULT_FINISH[tostring(cand.finishReason)] then
    return "done", finish_fault(cand)
  end
  local message = { role = "assistant", content = table.concat(out.text, "") }
  if #out.reasoning > 0 and not hide_reasoning then
    message.reasoning_content = table.concat(out.reasoning, "")
  end
  if #out.toolcalls > 0 then
    local calls = {}
    for _, tc in ipairs(out.toolcalls) do
      calls[#calls + 1] = openai_tool_call(tc, false)
    end
    message.tool_calls = calls
  end
  local finish = map_finish_reason(cand.finishReason)
  if #out.toolcalls > 0 then
    finish = "tool_calls"
  end
  return "ok",
    {
      id = (type(g.responseId) == "string" and g.responseId ~= "") and g.responseId or ("google-" .. random_id()),
      object = "chat.completion",
      created = os.time(),
      model = full_model,
      choices = { { index = cand.index or 0, message = message, finish_reason = finish } },
      usage = build_usage(g.usageMetadata) or { prompt_tokens = 0, completion_tokens = 0, total_tokens = 0 },
    }
end

-- First inlineData part of the given media family in any candidate.
local function find_inline_data(g, family_prefix)
  if type(g.candidates) ~= "table" then
    return nil
  end
  for _, cand in ipairs(g.candidates) do
    local parts = cand and cand.content and cand.content.parts or {}
    for _, p in ipairs(parts) do
      local inline = type(p) == "table" and p.inlineData or nil
      if
        inline
        and type(inline.data) == "string"
        and inline.data ~= ""
        and type(inline.mimeType) == "string"
        and inline.mimeType:sub(1, #family_prefix) == family_prefix
      then
        return inline
      end
    end
  end
  return nil
end

-- Outcome for a 200 that carried no media: a policy block ends the request
-- (the same prompt blocks on every key); anything else may be transient.
local function no_media_outcome(g, what)
  if type(g.promptFeedback) == "table" and g.promptFeedback.blockReason then
    return "done", content_filter_error(g.promptFeedback.blockReason)
  end
  local cand = type(g.candidates) == "table" and g.candidates[1] or nil
  if cand and POLICY_FINISH[tostring(cand.finishReason)] then
    return "done", error_of("gemini blocked the " .. what .. ": " .. tostring(cand.finishReason), "content_filter", 400)
  end
  return "retry_exit", error_of("upstream returned no generated " .. what, "server_error", 502)
end

-- RIFF/WAVE header around PCM s16le (the only encoding Gemini TTS emits).
-- PCM payload of a RIFF/WAVE file (its "data" chunk), or nil.
local function wav_data(raw)
  if raw:sub(1, 4) ~= "RIFF" or raw:sub(9, 12) ~= "WAVE" then
    return nil
  end
  local pos = 13
  while pos + 7 <= #raw do
    local b1, b2, b3, b4 = raw:byte(pos + 4, pos + 7)
    local size = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
    if raw:sub(pos, pos + 3) == "data" then
      return raw:sub(pos + 8, pos + 7 + size)
    end
    pos = pos + 8 + size + (size % 2)
  end
  return nil
end

local function wrap_wav(pcm, rate, channels)
  local bits = 16
  local function u32(n)
    return string.char(n % 256, math.floor(n / 256) % 256, math.floor(n / 65536) % 256, math.floor(n / 16777216) % 256)
  end
  local function u16(n)
    return string.char(n % 256, math.floor(n / 256) % 256)
  end
  local byte_rate = rate * channels * (bits / 8)
  local block_align = channels * (bits / 8)
  local data_size = #pcm
  return "RIFF"
    .. u32(36 + data_size)
    .. "WAVE"
    .. "fmt "
    .. u32(16)
    .. u16(1)
    .. u16(channels)
    .. u32(rate)
    .. u32(byte_rate)
    .. u16(block_align)
    .. u16(bits)
    .. "data"
    .. u32(data_size)
    .. pcm
end

-- Map an OpenAI size string ("1024x1024") to the nearest Gemini aspect ratio.
local ASPECT_RATIOS = {
  { ratio = 1.0, label = "1:1" },
  { ratio = 4 / 3, label = "4:3" },
  { ratio = 3 / 4, label = "3:4" },
  { ratio = 3 / 2, label = "3:2" },
  { ratio = 2 / 3, label = "2:3" },
  { ratio = 5 / 4, label = "5:4" },
  { ratio = 4 / 5, label = "4:5" },
  { ratio = 16 / 9, label = "16:9" },
  { ratio = 9 / 16, label = "9:16" },
  { ratio = 21 / 9, label = "21:9" },
}

local function aspect_ratio_of(size)
  if type(size) ~= "string" then
    return nil
  end
  local w, h = size:match("^(%d+)x(%d+)$")
  w, h = tonumber(w), tonumber(h)
  if not w or not h or w <= 0 or h <= 0 then
    return nil
  end
  local r = w / h
  local best, best_diff = nil, math.huge
  for _, c in ipairs(ASPECT_RATIOS) do
    local diff = math.abs(c.ratio - r)
    if diff < best_diff then
      best, best_diff = c.label, diff
    end
  end
  return best
end

-- Sniff an image mime from the first bytes of a base64 payload.
local function image_mime_of(b64, filename)
  local ok, raw = pcall(llm_router.base64_decode, b64:sub(1, 24))
  if ok and type(raw) == "string" then
    if raw:sub(1, 4) == "\137PNG" then
      return "image/png"
    end
    if raw:sub(1, 3) == "\255\216\255" then
      return "image/jpeg"
    end
    if raw:sub(1, 4) == "GIF8" then
      return "image/gif"
    end
    if raw:sub(1, 4) == "RIFF" and raw:sub(9, 12) == "WEBP" then
      return "image/webp"
    end
  end
  local ext = tostring(filename or ""):lower():match("%.(%w+)$")
  if ext == "jpg" or ext == "jpeg" then
    return "image/jpeg"
  end
  if ext == "webp" then
    return "image/webp"
  end
  if ext == "gif" then
    return "image/gif"
  end
  return "image/png"
end

-- Debug: was the native search tool sent, and did the model actually search?
local function log_search_request(env, request, payload)
  local sent = false
  for _, t in ipairs(payload.tools or {}) do
    if t.googleSearch ~= nil then
      sent = true
    end
  end
  local n = type(request.tools) == "table" and #request.tools or 0
  llm_router.log(
    "debug",
    "search request",
    { operation = env.op, model = env.model, request_tools = n, search_tool_sent = sent, event = "payload" }
  )
end

local function log_grounding_result(env, g)
  local cand = type(g) == "table" and type(g.candidates) == "table" and g.candidates[1] or nil
  local meta = cand and cand.groundingMetadata or nil
  local queries = meta and meta.webSearchQueries or nil
  local count, first = 0, ""
  if type(queries) == "table" then
    count = #queries
    first = trunc(queries[1], 120)
  end
  llm_router.log("debug", "grounding result", {
    operation = env.op,
    model = env.model,
    grounded = meta ~= nil,
    count = count,
    detail = first,
    event = "payload",
  })
end

-- ---------------------------------------------------------------------------
-- Failover runner
-- ---------------------------------------------------------------------------

-- A timed-out request is slow generation, not a bad exit: retrying it on the
-- next proxy only repeats the wait. The transport fault carries no timeout
-- flag, so judge by wording and by elapsed time against the client timeout.
local function is_timeout(fault, started, timeout_ms)
  local text = (tostring(fault.message or "") .. " " .. tostring(fault.reason or "")):lower()
  if has_any(text, { "timeout", "timed out", "deadline" }) then
    return true
  end
  return (os.time() - started) >= (math.floor(timeout_ms / 1000) - 1)
end

local function fault_reason(fault)
  if type(fault) ~= "table" then
    return tostring(fault)
  end
  return trunc(fault.reason or fault.message or "network error", 200)
end

-- Returns (error, retry_on_next_exit). Only proxied-leg faults the router
-- marks retryable move on to the next exit; a direct fault or a timeout would
-- fail identically on the next attempt.
local function classify_transport(env, fault, started, timeout_ms, cred_id, proxy_url)
  fault = type(fault) == "table" and fault or { message = tostring(fault) }
  local keys = {
    operation = env.op,
    model = env.model,
    cred_id = cred_id,
    reason = fault_reason(fault),
    proxy_url = redact_url(proxy_url),
    event = "http_error",
  }
  if is_timeout(fault, started, timeout_ms) then
    llm_router.log("error", "upstream request timed out", keys)
    return error_of(
      "gemini request timed out after " .. tostring(math.floor(timeout_ms / 1000)) .. "s",
      "server_error",
      504
    ),
      false
  end
  local err = error_of("gemini request failed: " .. fault_reason(fault), "server_error", 502)
  if fault.retryable == true then
    llm_router.log("warn", "exit failed", keys)
    return err, true
  end
  llm_router.log("error", "upstream request failed", keys)
  return err, false
end

-- Drives credentials x exits for one request. spec fields:
--   op, model (router id for logs), name (bare model), timeout_ms, exit_limit
--   attempt(env, client, key, proxy_url, cred) -> kind, value, tag
-- where kind is "ok" (value is the result), "done" (value is the terminal
-- error), "next_cred", "retry_exit" (value is the error to remember) or
-- "transport" (value is the http_client fault). Returns result, err.
local function run(ctx, spec)
  local env = { ctx = ctx, op = spec.op, model = spec.model, name = spec.name }
  local usable, total, min_wait, all_dead, all_free_tier = plan_credentials(spec.name)
  if all_dead then
    return nil, model_gone_error()
  end
  if #usable == 0 then
    if all_free_tier then
      return nil, free_tier_error()
    end
    if total > 0 then
      return nil, rate_limited_error(min_wait or DEFAULT_COOLDOWN_SECS)
    end
    return nil, error_of("no usable google credentials", "server_error", 503)
  end
  local client = llm_router.http_client({ timeout_ms = spec.timeout_ms })
  local last_err, attempts, bad_exits = nil, 0, {}
  for _, cred in ipairs(usable) do
    local key = api_key_of(cred.data)
    if key == "" then
      llm_router.log("warn", "credential has no api key", { operation = env.op, cred_id = cred.id })
      last_err = last_err or error_of("google credential has no api_key", "authentication_error", 401)
    else
      local exits, perr = pick_proxies(ctx, spec.exit_limit or 3, bad_exits)
      if not exits then
        return nil, perr
      end
      for _, px in ipairs(exits) do
        attempts = attempts + 1
        llm_router.log("debug", "trying credential", {
          operation = env.op,
          model = env.model,
          cred_id = cred.id,
          attempt = attempts,
          proxy_url = redact_url(px.url),
        })
        local started = os.time()
        local kind, value, tag = spec.attempt(env, client, key, px.url, cred)
        if kind == "ok" then
          llm_router.log(
            "debug",
            "request served",
            { operation = env.op, model = env.model, cred_id = cred.id, attempt = attempts }
          )
          return value, nil
        elseif kind == "done" then
          return nil, value
        elseif kind == "transport" then
          local terr, retry = classify_transport(env, value, started, spec.timeout_ms, cred.id, px.url)
          if not retry then
            return nil, terr
          end
          last_err = terr
          if px.url then
            bad_exits[#bad_exits + 1] = px.url
          end
        else
          last_err = value or last_err
          if kind == "next_cred" then
            break
          end
          if tag == "geo" and px.url then
            bad_exits[#bad_exits + 1] = px.url
          end
        end
      end
    end
  end
  local exhausted = last_err or error_of("all google credentials exhausted", "server_error", 502)
  llm_router.log("error", "all credentials exhausted", {
    operation = env.op,
    model = env.model,
    reason = trunc(exhausted.message or "request failed", 300),
    code = exhausted.code,
    count = attempts,
  })
  return nil, exhausted
end

local function post_json(client, url, key, payload, proxy_url)
  return client:request({
    method = "POST",
    url = url,
    headers = { ["x-goog-api-key"] = key, ["Content-Type"] = "application/json" },
    body = json.encode(payload),
    proxy_url = proxy_url,
  })
end

-- Shared unary attempt tail: transport fault, non-200 mapping, JSON decode,
-- then the endpoint's own parse(g) -> kind, value. An undecodable 200 (a
-- captive portal or a tampering free proxy) moves on to the next exit.
local function unary_outcome(env, resp, err, cred, parse)
  if err ~= nil then
    return "transport", err
  end
  if resp.status ~= 200 then
    return map_upstream(env, resp, cred.id)
  end
  local g = safe_decode(resp.body)
  if g == nil then
    llm_router.log("warn", "upstream returned an unparsable body", {
      operation = env.op,
      model = env.model,
      cred_id = cred.id,
      status = resp.status,
      body = trunc(resp.body, 300),
      event = "http_error",
    })
    return "retry_exit", error_of("gemini returned an unparsable response", "server_error", 502)
  end
  return parse(g)
end

-- ---------------------------------------------------------------------------
-- Metadata reads (model listing, key health, model existence)
-- ---------------------------------------------------------------------------

-- GET against the model catalogue. A configured pool carries every attempt:
-- a location block, a 5xx or a transport fault moves on to the next exit, and
-- nothing connects directly. An unconfigured provider makes one direct
-- attempt. Returns resp, err.
local function metadata_get(ctx, key, path)
  local client = llm_router.http_client({ timeout_ms = META_TIMEOUT_MS })
  local function get(proxy_url)
    return client:request({
      method = "GET",
      url = BASE_URL .. path,
      headers = { ["x-goog-api-key"] = key },
      proxy_url = proxy_url,
    })
  end
  if pool_of(ctx) == nil then
    return get(nil)
  end
  local exits, perr = pick_proxies(ctx, 3)
  if not exits then
    return nil, perr
  end
  local resp, err
  for _, px in ipairs(exits) do
    resp, err = get(px.url)
    if err == nil and not is_geo_block(resp.status, tostring(resp.body or ""):lower()) and resp.status < 500 then
      break
    end
  end
  return resp, err
end

-- Pages through /models. Returns entries, or nil plus {kind, resp|err}.
local function list_models(ctx, key)
  local entries, token, pages = {}, nil, 0
  repeat
    local path = "/models?pageSize=1000"
    if token then
      path = path .. "&pageToken=" .. url_encode(token)
    end
    local resp, err = metadata_get(ctx, key, path)
    if err ~= nil then
      return nil, { kind = "transport", err = err }
    end
    if resp.status ~= 200 then
      return nil, { kind = "status", resp = resp }
    end
    local page = safe_decode(resp.body)
    if page == nil then
      return nil, { kind = "body" }
    end
    for _, entry in ipairs(page.models or {}) do
      entries[#entries + 1] = entry
    end
    token = (type(page.nextPageToken) == "string" and page.nextPageToken ~= "") and page.nextPageToken or nil
    pages = pages + 1
  until token == nil or pages >= 20
  return entries, nil
end

-- ---------------------------------------------------------------------------
-- Model catalogue
-- ---------------------------------------------------------------------------

-- Advisory per-family rate defaults for the dashboard. Real limits depend on
-- the project's tier; override with model_specs when exact numbers matter.
local function estimate_rpm(name)
  local m = name:lower()
  if m:find("flash", 1, true) then
    return m:find("lite", 1, true) and 2000 or 1000
  end
  if m:find("pro", 1, true) then
    return 500
  end
  return 1000
end

local function estimate_tpm(name)
  local m = name:lower()
  if m:find("flash", 1, true) then
    return m:find("lite", 1, true) and 2000000 or 1000000
  end
  if m:find("pro", 1, true) then
    return 500000
  end
  return 1000000
end

local function estimate_rpd(name)
  local m = name:lower()
  if m:find("flash", 1, true) then
    return m:find("lite", 1, true) and 3000 or 1500
  end
  if m:find("pro", 1, true) then
    return 1000
  end
  return 1500
end

local function model_short_name(entry)
  if type(entry.name) == "string" and entry.name ~= "" then
    return bare_model(entry.name)
  end
  if type(entry.baseModelId) == "string" and entry.baseModelId ~= "" then
    return entry.baseModelId
  end
  return "unknown-model"
end

local function supports_method(entry, method)
  for _, m in ipairs(entry.supportedGenerationMethods or {}) do
    if m == method then
      return true
    end
  end
  return false
end

-- OpenRouter-style reasoning block of the model listing: level-mode models
-- advertise their efforts (see effort_levels), budget-mode models (Gemini 2.5)
-- advertise reasoning.max_tokens instead. Nil when the family does not reason.
local function reasoning_info(name)
  local mode = thinking_mode(name)
  if mode == "level" then
    return { default_enabled = true, supported_efforts = effort_list(name) }
  elseif mode == "budget" then
    return { default_enabled = true, supports_max_tokens = true }
  end
  return nil
end

local CHAT_PARAMETERS = {
  "tools",
  "tool_choice",
  "response_format",
  "structured_outputs",
  "temperature",
  "top_p",
  "top_k",
  "max_tokens",
  "seed",
  "stop",
  "reasoning",
  "reasoning_effort",
  "web_search",
}

-- Nil when the entry does not serve a method this plugin speaks.
local function model_info_of(entry)
  local name = model_short_name(entry)
  local lower = name:lower()
  local embedding = is_embedding_model(lower)
  local serves = supports_method(entry, "generateContent") or (embedding and supports_method(entry, "embedContent"))
  if not serves or lower:find("aqa", 1, true) then
    return nil
  end

  -- Gemini takes text, image, audio, video and file (document) input; gemma-4
  -- takes images. Output stays text unless the family says otherwise: chat
  -- completions here never request IMAGE/AUDIO response modalities.
  local in_modalities = { "text" }
  if lower:match("^gemini") then
    in_modalities = { "text", "image", "audio", "video", "file" }
  elseif lower:match("^gemma") then
    in_modalities = { "text", "image" }
  end
  local out_modalities = { "text" }
  local endpoints = nil
  local params = CHAT_PARAMETERS
  if embedding then
    in_modalities, out_modalities, endpoints, params = { "text" }, { "embedding" }, { "embeddings" }, { "dimensions" }
  elseif is_tts_model(lower) then
    in_modalities, out_modalities, endpoints, params = { "text" }, { "audio" }, { "audio/speech" }, { "temperature" }
  elseif is_transcribe_model(lower) then
    in_modalities, out_modalities = { "audio" }, { "text" }
    endpoints, params = { "audio/transcriptions", "audio/translations" }, { "temperature" }
  elseif is_image_model(lower) then
    out_modalities = { "text", "image" }
    endpoints = { "chat/completions", "images/generations" }
  end

  local display = entry.displayName
  if display == nil or display == "" then
    display = name
  end
  local info = {
    name = name,
    display_name = display,
    rpm = estimate_rpm(name),
    tpm = estimate_tpm(name),
    rpd = estimate_rpd(name),
    context_window = entry.inputTokenLimit or 0,
    max_tokens = entry.outputTokenLimit or 0,
    input_modalities = in_modalities,
    output_modalities = out_modalities,
    supported_parameters = params,
  }
  if endpoints then
    info.endpoints = endpoints
  end
  if entry.thinking == true or is_gemma(lower) then
    info.reasoning = reasoning_info(name)
  end
  return info
end

-- ---------------------------------------------------------------------------
-- Model probes
-- ---------------------------------------------------------------------------

-- Generation method each probed endpoint rides on.
local PROBE_METHODS = {
  chat = "generateContent",
  vision = "generateContent",
  speech = "generateContent",
  image = "generateContent",
  transcription = "generateContent",
  embeddings = "embedContent",
}

-- A model class that cannot serve the endpoint is ruled out before any request.
local function endpoint_mismatch(endpoint, name)
  if endpoint == "speech" then
    return not is_tts_model(name)
  end
  if endpoint == "image" then
    return not is_image_model(name)
  end
  if endpoint == "embeddings" then
    return not is_embedding_model(name)
  end
  if endpoint == "transcription" then
    return is_tts_model(name) or is_embedding_model(name)
  end
  return is_tts_model(name) or is_embedding_model(name)
end

local function probe_request(endpoint, name)
  local url = generate_url(name)
  if endpoint == "embeddings" then
    return BASE_URL .. "/models/" .. name .. ":embedContent",
      { model = "models/" .. name, content = { parts = { { text = "probe" } } } }
  elseif endpoint == "speech" then
    return url,
      {
        contents = { { role = "user", parts = { { text = "hi" } } } },
        generationConfig = {
          responseModalities = { "AUDIO" },
          speechConfig = { voiceConfig = { prebuiltVoiceConfig = { voiceName = "Kore" } } },
        },
      }
  elseif endpoint == "image" then
    return url,
      {
        contents = { { role = "user", parts = { { text = "a dot" } } } },
        generationConfig = { responseModalities = { "TEXT", "IMAGE" } },
      }
  elseif endpoint == "transcription" then
    -- One second of 16 kHz silence.
    local silence = llm_router.base64_encode(wrap_wav(string.rep("\0", 32000), 16000, 1))
    return url,
      {
        contents = {
          {
            role = "user",
            parts = { { inlineData = { mimeType = "audio/wav", data = silence } }, { text = "Transcribe this audio." } },
          },
        },
      }
  end
  return url,
    {
      contents = { { role = "user", parts = { { text = "Reply with: ok" } } } },
      generationConfig = { maxOutputTokens = 32 },
    }
end

-- Kinds: quota (429), missing (404), geo, provider (5xx), request (a 400 our
-- probe or a capability mismatch caused), key (rejected), other.
local function classify_probe_failure(status, body)
  local lower = tostring(body or ""):lower()
  if is_free_tier_block(status, lower) then
    return { kind = "free_tier", message = "model is not available on the free tier" }
  end
  if is_interactions_only(status, lower) then
    return { kind = "interactions", message = "model works only through the Interactions API" }
  end
  if status == 429 then
    return { kind = "quota", message = "rate limited" }
  end
  if is_key_rejection(status, lower) or status == 403 then
    return { kind = "key", message = "api key rejected" }
  end
  if status == 404 then
    return { kind = "missing", message = "model not found upstream" }
  end
  if is_geo_block(status, lower) then
    return { kind = "geo", message = "location blocked on exit" }
  end
  if status == 400 or status == 422 then
    return { kind = "request", message = "gemini rejected the probe request" }
  end
  if status >= 500 then
    return { kind = "provider", message = "gemini returned status " .. tostring(status) }
  end
  return { kind = "other", message = "request failed with status " .. tostring(status) }
end

-- One real generation for one key. Never parks, benches, disables or writes a
-- cooldown. Returns true, or nil plus {kind, message}.
local function probe_generation(ctx, key, name, endpoint)
  local url, payload = probe_request(endpoint, name)
  local client = llm_router.http_client({ timeout_ms = PROBE_TIMEOUT_MS })
  local exits, perr = pick_proxies(ctx, 2)
  if not exits then
    return nil, { kind = "other", message = perr.message }
  end
  local last = { kind = "other", message = "probe failed" }
  for _, px in ipairs(exits) do
    local resp, err = post_json(client, url, key, payload, px.url)
    if err ~= nil then
      last = { kind = "transport", message = fault_reason(err) }
      if type(err) ~= "table" or err.retryable ~= true then
        return nil, last
      end
    elseif resp.status == 200 then
      return true
    else
      last = classify_probe_failure(resp.status, resp.body)
      if last.kind ~= "geo" and last.kind ~= "provider" then
        return nil, last
      end
    end
  end
  return nil, last
end

-- Real availability, judged by actual generation. The cheap metadata listing
-- rules a model out first (missing upstream, or the method is not offered),
-- then a minimal generation runs on each usable key and the first success
-- wins. Quota outranks every failure; death needs a 404 or unanimous
-- provider-side failure; anything else is inconclusive.
local function check_model(ctx, model_ref)
  local endpoint = type(model_ref.endpoint) == "string" and model_ref.endpoint ~= "" and model_ref.endpoint or "chat"
  local want_method = PROBE_METHODS[endpoint]
  if not want_method then
    return { status = "unknown", message = "endpoint not supported: " .. endpoint }
  end
  local name = bare_model(model_ref.model_name or "")
  if name == "" then
    return { status = "unknown", message = "model name is required" }
  end
  if endpoint_mismatch(endpoint, name) then
    return { status = "unknown", message = "endpoint not supported for this model: " .. endpoint }
  end

  local usable, total, _, all_dead, all_free_tier = plan_credentials(name)
  if all_dead then
    return { status = "unhealthy", message = "model no longer available to these keys" }
  end
  if #usable == 0 and all_free_tier then
    return { status = "unhealthy", message = "model is not available on the free tier of these keys" }
  end
  if #usable == 0 then
    if total > 0 then
      return { status = "quota_exceeded", message = "quota exhausted" }
    end
    return { status = "unknown", message = "no credentials available" }
  end

  local seen_unknown = nil
  local tried, missing, unsupported, listed = 0, 0, 0, false
  for _, cred in ipairs(usable) do
    local key = api_key_of(cred.data)
    if key == "" then
      seen_unknown = seen_unknown or "api_key is required"
    else
      tried = tried + 1
      llm_router.log(
        "debug",
        "probing model metadata",
        { operation = "check_model", cred_id = cred.id, model = model_ref.model, endpoint = endpoint }
      )
      local resp, err = metadata_get(ctx, key, "/models/" .. url_encode(name))
      if err ~= nil then
        seen_unknown = seen_unknown or fault_reason(err)
      elseif resp.status == 200 then
        local meta = safe_decode(resp.body)
        if meta == nil then
          seen_unknown = seen_unknown or "unparsable model listing"
        elseif supports_method(meta, want_method) then
          listed = true
          break
        else
          unsupported = unsupported + 1
        end
      elseif resp.status == 404 then
        missing = missing + 1
      elseif is_key_rejection(resp.status, tostring(resp.body or ""):lower()) or resp.status == 403 then
        seen_unknown = seen_unknown or "api key rejected"
      else
        seen_unknown = seen_unknown or ("status " .. tostring(resp.status))
      end
    end
  end
  if not listed then
    if tried > 0 and missing == tried then
      return { status = "unhealthy", message = "model not found upstream" }
    end
    if tried > 0 and missing + unsupported == tried and unsupported > 0 then
      return { status = "unknown", message = "endpoint not supported: " .. want_method .. " not offered by the model" }
    end
  end

  local quota_seen, missing_seen = false, false
  local generations, provider_failures, free_tier = 0, 0, 0
  for _, cred in ipairs(usable) do
    local key = api_key_of(cred.data)
    if key ~= "" then
      generations = generations + 1
      llm_router.log(
        "debug",
        "probing model generation",
        { operation = "check_model", cred_id = cred.id, model = model_ref.model, endpoint = endpoint }
      )
      local ok, failure = probe_generation(ctx, key, name, endpoint)
      if ok then
        return { status = "healthy", response = "ok" }
      end
      if failure.kind == "interactions" then
        return { status = "unhealthy", message = "model works only through the Interactions API" }
      elseif failure.kind == "free_tier" then
        free_tier = free_tier + 1
      elseif failure.kind == "quota" then
        quota_seen = true
      elseif failure.kind == "missing" then
        missing_seen = true
      elseif failure.kind == "provider" then
        provider_failures = provider_failures + 1
      else
        seen_unknown = seen_unknown or failure.message
      end
    end
  end
  if generations > 0 and free_tier == generations then
    return { status = "unhealthy", message = "model is not available on the free tier of these keys" }
  end
  if quota_seen then
    return { status = "quota_exceeded", message = "quota exhausted" }
  end
  if missing_seen then
    return { status = "unhealthy", message = "model not found upstream" }
  end
  if generations > 0 and provider_failures == generations then
    return { status = "unhealthy", message = "provider failed upstream" }
  end
  return { status = "unknown", message = seen_unknown or "health check inconclusive" }
end

-- ---------------------------------------------------------------------------
-- Embeddings helper
-- ---------------------------------------------------------------------------

-- Truncated Matryoshka vectors are not unit length; normalising is idempotent
-- for vectors that already are.
local function normalize_vector(values)
  local sum = 0
  for _, v in ipairs(values) do
    sum = sum + v * v
  end
  if sum <= 0 then
    return values
  end
  local norm = math.sqrt(sum)
  local out = {}
  for i, v in ipairs(values) do
    out[i] = v / norm
  end
  return out
end

local AUDIO_MIME_ALIASES = {
  ["audio/mpeg"] = "audio/mp3",
  ["audio/x-wav"] = "audio/wav",
  ["audio/wave"] = "audio/wav",
  ["audio/vnd.wave"] = "audio/wav",
  ["audio/x-flac"] = "audio/flac",
  ["audio/x-aiff"] = "audio/aiff",
}
local AUDIO_MIME_BY_EXTENSION = {
  wav = "audio/wav",
  mp3 = "audio/mp3",
  aiff = "audio/aiff",
  aif = "audio/aiff",
  aac = "audio/aac",
  ogg = "audio/ogg",
  oga = "audio/ogg",
  flac = "audio/flac",
}

local function audio_mime_of(content_type, filename)
  local mime = tostring(content_type or ""):lower():match("^([^;%s]+)")
  if mime and mime:sub(1, 6) == "audio/" then
    return AUDIO_MIME_ALIASES[mime] or mime
  end
  local ext = tostring(filename or ""):lower():match("%.(%w+)$")
  return ext and AUDIO_MIME_BY_EXTENSION[ext] or nil
end

-- ---------------------------------------------------------------------------
-- Registration
-- ---------------------------------------------------------------------------

llm_router.register("google", {
  icon = "https://www.gstatic.com/lamda/images/favicon_v1_150160cddff7f294ce30.svg",

  proxy_schema = {},

  locales = {
    ["en"] = {
      ["google.subtitle"] = "Get your API key from Google AI Studio.",
      ["google.open_studio"] = "Open AI Studio",
      ["misc.api_key"] = "API Key",
      ["misc.save"] = "Save",
    },
    ["ru"] = {
      ["google.subtitle"] = "Получите API-ключ в Google AI Studio.",
      ["google.open_studio"] = "Открыть AI Studio",
      ["misc.api_key"] = "API-ключ",
      ["misc.save"] = "Сохранить",
    },
  },

  credential_schema = {
    {
      type = "section",
      title = "Google AI Studio",
      subtitle = "google.subtitle",
      content = {
        { type = "link", text = "google.open_studio", url = "https://aistudio.google.com/app/apikey" },
        { type = "secret", name = "api_key", label = "misc.api_key", required = true },
        { type = "button", text = "misc.save", form_action = "submit" },
      },
    },
  },

  validate_credentials = function(data)
    local key = type(data) == "table" and data.api_key or nil
    if type(key) ~= "string" or key == "" then
      return false, invalid_request("api_key is required")
    end
    if #key < 20 then
      return false, invalid_request("api_key: minimum 20 characters")
    end
    if key:find("%s") then
      return false, invalid_request("api_key must not contain whitespace")
    end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, invalid_request("no google credentials configured")
    end
    local entries, failure = nil, nil
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key ~= "" then
        local found, why = list_models(ctx, key)
        if found then
          entries = found
          break
        end
        failure = why
      end
    end
    if entries == nil then
      if failure == nil then
        return nil, invalid_request("google credentials carry no api_key")
      end
      if failure.kind == "status" then
        local status = failure.resp.status
        llm_router.log(
          "error",
          "discovery rejected",
          { operation = "get_model_infos", status = status, body = trunc(failure.resp.body, 500) }
        )
        return nil,
          error_of(
            "google model discovery failed with status " .. tostring(status),
            "server_error",
            (status >= 400 and status < 600) and status or 502
          )
      elseif failure.kind == "transport" then
        llm_router.log(
          "error",
          "discovery transport error",
          { operation = "get_model_infos", reason = fault_reason(failure.err) }
        )
        return nil, error_of("google model discovery failed: " .. fault_reason(failure.err), "server_error", 502)
      end
      return nil, error_of("google model discovery returned an unparsable page", "server_error", 502)
    end
    local infos = {}
    for _, entry in ipairs(entries) do
      local info = model_info_of(entry)
      if info and not dead_everywhere(creds, info.name) then
        infos[#infos + 1] = info
      end
    end
    return infos
  end,

  -- Account liveness: the keyed /models listing answers 200 for live keys.
  -- Only explicit key-rejection signals verdict unhealthy; location blocks,
  -- quotas and every other failure are unknown and change nothing.
  check_health = function(ctx, credential)
    local key = api_key_of(credential and credential.data)
    if key == "" then
      return { status = "unhealthy", message = "api_key is required" }
    end
    local resp, err = metadata_get(ctx, key, "/models?pageSize=1")
    if err ~= nil then
      return { status = "unknown", message = fault_reason(err) }
    end
    if resp.status == 200 then
      return { status = "healthy" }
    end
    if is_key_rejection(resp.status, tostring(resp.body or ""):lower()) then
      return { status = "unhealthy", message = "api key rejected" }
    end
    return { status = "unknown", message = "status " .. tostring(resp.status) }
  end,

  check_model = check_model,

  complete = function(ctx, request)
    local payload, perr = build_payload(request, ctx.provider_config)
    if perr then
      return nil, perr
    end
    local name = bare_model(request.model_name)
    local sig_in_id = sig_in_id_of(ctx.provider_config)
    log_search_request({ op = "complete", model = request.model }, request, payload)
    local url = generate_url(name)
    local result, err = run(ctx, {
      op = "complete",
      model = request.model,
      name = name,
      timeout_ms = CHAT_TIMEOUT_MS,
      attempt = function(env, client, key, proxy_url, cred)
        local resp, call_err = post_json(client, url, key, payload, proxy_url)
        return unary_outcome(env, resp, call_err, cred, function(g)
          log_grounding_result(env, g)
          return assemble_completion(g, request.model, sig_in_id, reasoning_excluded(request))
        end)
      end,
    })
    if err then
      return nil, err
    end
    return result, nil
  end,

  -- Streams belong to the attempt that emitted first: before the first emit
  -- the runner may fail over to another exit or key; afterwards a failure is
  -- final. Usage is buffered and sent once, because Gemini repeats cumulative
  -- usageMetadata on every event.
  complete_stream = function(ctx, request, emit)
    local payload, perr = build_payload(request, ctx.provider_config)
    if perr then
      return nil, perr
    end
    local name = bare_model(request.model_name)
    local full_model = request.model
    local sig_in_id = sig_in_id_of(ctx.provider_config)
    local hide_reasoning = reasoning_excluded(request)
    local chunk_id = "google-" .. random_id()
    local url = BASE_URL .. "/models/" .. name .. ":streamGenerateContent?alt=sse"
    log_search_request({ op = "complete_stream", model = full_model }, request, payload)

    local _, err = run(ctx, {
      op = "complete_stream",
      model = full_model,
      name = name,
      timeout_ms = STREAM_TIMEOUT_MS,
      attempt = function(env, client, key, proxy_url, cred)
        local started = os.time()
        local st = {
          seq = { n = 0 },
          emitted = false,
          role_sent = false,
          saw_tools = false,
          finish_sent = false,
          stopped = false,
        }

        local function send(choice, usage)
          local chunk = {
            id = st.id or chunk_id,
            object = "chat.completion.chunk",
            created = os.time(),
            model = full_model,
            choices = choice and { choice } or {},
          }
          if usage then
            chunk.usage = usage
          end
          emit(chunk)
          st.emitted = true
        end

        local function on_line(line)
          if st.stopped then
            return
          end
          local data = line:match("^data:%s?(.*)$")
          if data == nil or data == "" or data == "[DONE]" then
            return
          end
          local g = safe_decode(data)
          if g == nil then
            return
          end
          if type(g.error) == "table" then
            st.stopped = true
            if st.emitted then
              st.mid_error = error_of(
                "gemini stream reported an error: " .. trunc(g.error.message or g.error.status or "unknown", 200),
                "server_error",
                502
              )
            else
              st.soft = { status = tonumber(g.error.code) or 502, body = data, headers = {} }
            end
            return
          end
          if type(g.responseId) == "string" and g.responseId ~= "" and not st.emitted then
            st.id = g.responseId
          end
          if type(g.promptFeedback) == "table" and g.promptFeedback.blockReason then
            st.block = tostring(g.promptFeedback.blockReason)
          end
          if type(g.usageMetadata) == "table" then
            st.usage = build_usage(g.usageMetadata)
          end
          local cand = type(g.candidates) == "table" and g.candidates[1] or nil
          if cand == nil then
            return
          end
          if cand.groundingMetadata and not st.grounded then
            st.grounded = true
            log_grounding_result(env, g)
          end
          local out = { text = {}, reasoning = {}, toolcalls = {} }
          collect_parts(cand, out, st.seq, sig_in_id)
          local finish_raw = type(cand.finishReason) == "string" and cand.finishReason or ""
          if
            FAULT_FINISH[finish_raw]
            and not st.emitted
            and #out.text == 0
            and #out.reasoning == 0
            and #out.toolcalls == 0
          then
            st.fault = finish_fault(cand)
            st.stopped = true
            return
          end
          local delta = {}
          if #out.reasoning > 0 and not hide_reasoning then
            delta.reasoning_content = table.concat(out.reasoning, "")
          end
          if #out.text > 0 then
            delta.content = table.concat(out.text, "")
          end
          if #out.toolcalls > 0 then
            st.saw_tools = true
            local calls = {}
            for _, tc in ipairs(out.toolcalls) do
              calls[#calls + 1] = openai_tool_call(tc, true)
            end
            delta.tool_calls = calls
          end
          local finish = nil
          if finish_raw ~= "" then
            finish = map_finish_reason(finish_raw)
            -- Gemini ends a tool-calling turn with a bare STOP: run the tools.
            if st.saw_tools and finish == "stop" then
              finish = "tool_calls"
            end
            st.finish_sent = true
          end
          if next(delta) == nil and finish == nil then
            return
          end
          if not st.role_sent then
            delta.role = "assistant"
            st.role_sent = true
          end
          send({ index = cand.index or 0, delta = delta, finish_reason = finish })
        end

        local _, stream_err = client:stream({
          method = "POST",
          url = url,
          headers = { ["x-goog-api-key"] = key, ["Content-Type"] = "application/json" },
          body = json.encode(payload),
          proxy_url = proxy_url,
          on_response = function(r)
            if r.status == 200 then
              return nil
            end
            local action, terr, tag = map_upstream(env, r, cred.id)
            st.action, st.terr, st.tag = action, terr, tag
            return terr
          end,
          on_line = on_line,
        })

        if st.action ~= nil then
          return st.action, st.terr, st.tag
        end
        if stream_err ~= nil then
          if st.emitted then
            local fault = type(stream_err) == "table" and stream_err or { message = tostring(stream_err) }
            llm_router.log("error", "stream failed after output", {
              operation = env.op,
              model = env.model,
              cred_id = cred.id,
              reason = fault_reason(fault),
              event = "stream_error",
            })
            if is_timeout(fault, started, STREAM_TIMEOUT_MS) then
              return "done",
                error_of(
                  "gemini stream timed out after " .. tostring(math.floor(STREAM_TIMEOUT_MS / 1000)) .. "s",
                  "server_error",
                  504
                )
            end
            return "done", error_of("gemini stream interrupted: " .. fault_reason(fault), "server_error", 502)
          end
          return "transport", stream_err
        end
        if st.mid_error then
          llm_router.log(
            "error",
            "stream reported an error after output",
            { operation = env.op, model = env.model, cred_id = cred.id, event = "stream_error" }
          )
          return "done", st.mid_error
        end
        if not st.emitted then
          if st.soft then
            return map_upstream(env, st.soft, cred.id)
          end
          if st.block then
            return "done", content_filter_error(st.block)
          end
          if st.fault then
            return "done", st.fault
          end
          return "retry_exit", error_of("upstream returned no generated content", "server_error", 502)
        end
        if not st.finish_sent then
          send({ index = 0, delta = {}, finish_reason = st.saw_tools and "tool_calls" or "stop" })
        end
        if st.usage then
          send(nil, st.usage)
        end
        llm_router.log(
          "debug",
          "stream completed",
          { operation = env.op, model = env.model, cred_id = cred.id, event = "stream_done" }
        )
        return "ok", true
      end,
    })
    if err then
      return nil, err
    end
  end,

  speech = function(ctx, request)
    if type(request.input) ~= "string" or request.input == "" then
      return nil, invalid_request("input is required")
    end
    local voice = request.voice
    if type(voice) ~= "string" or voice == "" then
      voice = "Kore"
    end
    -- Style control on Gemini TTS is prompt-based: "Say cheerfully: ...".
    local text = request.input
    if type(request.instructions) == "string" and request.instructions ~= "" then
      text = request.instructions .. ": " .. text
    end
    local payload = {
      contents = { { role = "user", parts = { { text = text } } } },
      generationConfig = {
        responseModalities = { "AUDIO" },
        speechConfig = { voiceConfig = { prebuiltVoiceConfig = { voiceName = voice } } },
      },
    }
    local name = bare_model(request.model_name)
    local url = generate_url(name)
    local requested = type(request.response_format) == "string" and request.response_format:lower() or ""
    local result, err = run(ctx, {
      op = "speech",
      model = request.model,
      name = name,
      timeout_ms = MEDIA_TIMEOUT_MS,
      attempt = function(env, client, key, proxy_url, cred)
        local resp, call_err = post_json(client, url, key, payload, proxy_url)
        return unary_outcome(env, resp, call_err, cred, function(g)
          local inline = find_inline_data(g, "audio/")
          if not inline then
            return no_media_outcome(g, "audio")
          end
          -- Upstream audio is raw PCM s16le ("audio/L16;...;rate=24000") on
          -- older TTS models and a finished WAV ("audio/wav") from 3.8 on.
          -- Only pcm and wav are servable without transcoding; every other
          -- requested format gets wav, the lossless playable container.
          local mime = tostring(inline.mimeType or ""):lower()
          local decoded, raw = pcall(llm_router.base64_decode, inline.data)
          if not decoded then
            return "retry_exit", error_of("upstream returned undecodable audio", "server_error", 502)
          end
          if raw:sub(1, 4) == "RIFF" then
            if requested ~= "pcm" then
              return "ok", { audio_b64 = inline.data, format = "wav" }
            end
            local pcm = wav_data(raw)
            if not pcm then
              return "retry_exit", error_of("upstream returned a malformed wav", "server_error", 502)
            end
            return "ok", { audio_b64 = llm_router.base64_encode(pcm), format = "pcm" }
          end
          if requested == "pcm" then
            return "ok", { audio_b64 = inline.data, format = "pcm" }
          end
          local rate = tonumber(mime:match("rate=(%d+)") or "24000") or 24000
          return "ok", { audio_b64 = llm_router.base64_encode(wrap_wav(raw, rate, 1)), format = "wav" }
        end)
      end,
    })
    if err then
      return nil, err
    end
    return result, nil
  end,

  transcribe = function(ctx, request)
    local file = request.file
    if type(file) ~= "string" or file == "" then
      return nil, invalid_request("file is required")
    end
    if #file > MAX_INLINE_AUDIO_BYTES then
      return nil, invalid_request("audio exceeds the 14 MB inline limit; split it into shorter files")
    end
    local format = type(request.response_format) == "string" and request.response_format:lower() or "json"
    if request.needs_segments == true or format == "srt" or format == "vtt" or format == "verbose_json" then
      return nil,
        invalid_request(
          "response_format "
            .. format
            .. " needs timestamps, which gemini transcription does not provide; use json or text"
        )
    end
    local mime = audio_mime_of(request.content_type, request.file_name)
    if not mime then
      return nil, invalid_request("unsupported audio type; use wav, mp3, aiff, aac, ogg or flac")
    end
    local instruction
    if request.task == "translate" then
      instruction = "Translate the speech in this audio into English. Output only the translation."
    else
      instruction =
        "Transcribe the speech in this audio verbatim. Output only the transcript text, with no commentary, speaker labels or timestamps."
    end
    if type(request.language) == "string" and request.language ~= "" then
      instruction = instruction .. " The spoken language is " .. request.language .. " (ISO 639-1)."
    end
    if type(request.prompt) == "string" and request.prompt ~= "" then
      instruction = instruction .. " Context: " .. request.prompt
    end
    local payload = {
      contents = {
        {
          role = "user",
          parts = {
            { inlineData = { mimeType = mime, data = llm_router.base64_encode(file) } },
            { text = instruction },
          },
        },
      },
    }
    if type(request.temperature) == "number" and request.temperature >= 0 then
      payload.generationConfig = { temperature = math.min(request.temperature, 2) }
    end
    local name = bare_model(request.model_name)
    local url = generate_url(name)
    local result, err = run(ctx, {
      op = "transcribe",
      model = request.model,
      name = name,
      timeout_ms = MEDIA_TIMEOUT_MS,
      attempt = function(env, client, key, proxy_url, cred)
        local resp, call_err = post_json(client, url, key, payload, proxy_url)
        return unary_outcome(env, resp, call_err, cred, function(g)
          local cand = type(g.candidates) == "table" and g.candidates[1] or nil
          local out = { text = {}, reasoning = {}, toolcalls = {} }
          if cand then
            collect_parts(cand, out, { n = 0 }, false)
          end
          local text = table.concat(out.text, ""):match("^%s*(.-)%s*$")
          if text == "" and cand and type(cand.content) == "table" then
            -- Transcribe models answer with audioTranscription parts, not text parts.
            local spoken = {}
            for _, part in ipairs(cand.content.parts or {}) do
              if type(part.audioTranscription) == "table" and type(part.audioTranscription.text) == "string" then
                spoken[#spoken + 1] = part.audioTranscription.text
              end
            end
            text = table.concat(spoken, ""):match("^%s*(.-)%s*$")
          end
          if text == "" then
            return no_media_outcome(g, "transcript")
          end
          return "ok", { text = text }
        end)
      end,
    })
    if err then
      return nil, err
    end
    return result, nil
  end,

  generate_image = function(ctx, request)
    if type(request.prompt) ~= "string" or request.prompt == "" then
      return nil, invalid_request("prompt is required")
    end
    if type(request.mask_b64) == "string" and request.mask_b64 ~= "" then
      return nil, invalid_request("mask_b64 is not supported by gemini image models; describe the edit in the prompt")
    end
    local n = 1
    if type(request.n) == "number" and request.n >= 1 then
      n = math.min(math.floor(request.n), MAX_IMAGES_PER_CALL)
    end
    local parts = {}
    -- Edits and variations: the source image rides in front of the prompt.
    if type(request.image_b64) == "string" and request.image_b64 ~= "" then
      parts[#parts + 1] =
        { inlineData = { mimeType = image_mime_of(request.image_b64, request.image_name), data = request.image_b64 } }
    end
    parts[#parts + 1] = { text = request.prompt }
    local gen_config = { responseModalities = { "TEXT", "IMAGE" } }
    local ratio = aspect_ratio_of(request.size)
    if ratio then
      gen_config.imageConfig = { aspectRatio = ratio }
    end
    local payload = { contents = { { role = "user", parts = parts } }, generationConfig = gen_config }
    local name = bare_model(request.model_name)
    local url = generate_url(name)
    local result, err = run(ctx, {
      op = "generate_image",
      model = request.model,
      name = name,
      timeout_ms = MEDIA_TIMEOUT_MS,
      attempt = function(env, client, key, proxy_url, cred)
        -- One upstream call per requested image: image models do not take
        -- candidateCount. A failed batch restarts on the next attempt; partial
        -- images are discarded, never half-returned.
        local data = {}
        for _ = 1, n do
          local resp, call_err = post_json(client, url, key, payload, proxy_url)
          local kind, value, tag = unary_outcome(env, resp, call_err, cred, function(g)
            local inline = find_inline_data(g, "image/")
            if not inline then
              return no_media_outcome(g, "image")
            end
            return "ok", { b64_json = inline.data }
          end)
          if kind ~= "ok" then
            return kind, value, tag
          end
          data[#data + 1] = value
        end
        return "ok", { created = os.time(), data = data }
      end,
    })
    if err then
      return nil, err
    end
    return result, nil
  end,

  embed = function(ctx, request)
    local input = request.input
    if type(input) == "string" then
      input = { input }
    end
    if type(input) ~= "table" or #input == 0 then
      return nil, invalid_request("input is required")
    end
    local name = bare_model(request.model_name)
    local dimensions = nil
    if type(request.dimensions) == "number" and request.dimensions > 0 then
      dimensions = math.floor(request.dimensions)
    end
    local requests = {}
    for _, text in ipairs(input) do
      if type(text) ~= "string" or text == "" then
        return nil, invalid_request("input must not contain empty strings")
      end
      local entry = { model = "models/" .. name, content = { parts = { { text = text } } } }
      if dimensions then
        entry.outputDimensionality = dimensions
      end
      requests[#requests + 1] = entry
    end
    local url = BASE_URL .. "/models/" .. name .. ":batchEmbedContents"
    local result, err = run(ctx, {
      op = "embed",
      model = request.model,
      name = name,
      timeout_ms = EMBED_TIMEOUT_MS,
      attempt = function(env, client, key, proxy_url, cred)
        local data = {}
        for first = 1, #requests, EMBED_BATCH_LIMIT do
          local batch = {}
          for i = first, math.min(first + EMBED_BATCH_LIMIT - 1, #requests) do
            batch[#batch + 1] = requests[i]
          end
          local resp, call_err = post_json(client, url, key, { requests = batch }, proxy_url)
          local kind, value, tag = unary_outcome(env, resp, call_err, cred, function(g)
            local embeddings = g.embeddings
            if type(embeddings) ~= "table" or #embeddings ~= #batch then
              return "retry_exit",
                error_of("upstream returned a different number of embeddings than requested", "server_error", 502)
            end
            local vectors = {}
            for _, item in ipairs(embeddings) do
              if type(item) ~= "table" or type(item.values) ~= "table" or #item.values == 0 then
                return "done", error_of("upstream returned an empty embedding vector", "server_error", 502)
              end
              vectors[#vectors + 1] = dimensions and normalize_vector(item.values) or item.values
            end
            return "ok", vectors
          end)
          if kind ~= "ok" then
            return kind, value, tag
          end
          for _, vector in ipairs(value) do
            data[#data + 1] = { object = "embedding", index = #data, embedding = vector }
          end
        end
        return "ok", { object = "list", data = data }
      end,
    })
    if err then
      return nil, err
    end
    return result, nil
  end,
})
