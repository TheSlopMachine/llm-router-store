--- @plugin OpenRouter
--- @author llm-router
--- @version 1.1.1
--- @plugin_api 1.6
--- @allow_host openrouter.ai
--- @description OpenRouter API: chat (unary and streaming), embeddings, speech-to-text, text-to-speech, image and video generation, model discovery and health probes

-- OpenRouter's OpenAI-compatible chat plus its media endpoints. Request and
-- response shapes follow the OpenRouter API reference.

local BASE_URL = "https://openrouter.ai/api/v1"

local UNARY_TIMEOUT_MS = 300000
local MEDIA_TIMEOUT_MS = 300000
local STREAM_TIMEOUT_MS = 900000
local DISCOVERY_TIMEOUT_MS = 60000
local PROBE_TIMEOUT_MS = 15000
local PROBE_GENERATION_TIMEOUT_MS = 60000

-- Cooldowns in seconds. Park is for transient states, bench for rejected keys.
local PARK_SECS = 300
local RATE_LIMIT_SECS = 60
local BENCH_SECS = 300
local LIFETIME_BUDGET_SECS = 86400
local MAX_PARK_SECS = 604800

-- Model `endpoints` paths this plugin serves, mapped to router endpoint names.
-- Paths it does not serve (messages, responses, legacy completions) are dropped.
local ENDPOINT_NAMES = {
  ["/api/v1/chat/completions"] = "chat/completions",
  ["/api/v1/embeddings"] = "embeddings",
  ["/api/v1/audio/speech"] = "audio/speech",
  ["/api/v1/audio/transcriptions"] = "audio/transcriptions",
  ["/api/v1/images"] = "images/generations",
  ["/api/v1/images/generations"] = "images/generations",
  ["/api/v1/videos"] = "videos",
}

-- Output modalities a model must advertise for each capability check_model probes.
local CAPABILITY_OUTPUT = {
  ["chat"] = { "text" },
  ["chat/completions"] = { "text" },
  ["vision"] = { "text" },
  ["embeddings"] = { "embeddings" },
  ["speech"] = { "speech", "audio" },
  ["audio/speech"] = { "speech", "audio" },
  ["transcription"] = { "transcription" },
  ["audio/transcriptions"] = { "transcription" },
  ["image"] = { "image" },
  ["images/generations"] = { "image" },
  ["video"] = { "video" },
  ["videos"] = { "video" },
}

local AUDIO_TYPES = {
  ["audio/wav"] = "wav",
  ["audio/x-wav"] = "wav",
  ["audio/wave"] = "wav",
  ["audio/mpeg"] = "mp3",
  ["audio/mp3"] = "mp3",
  ["audio/mp4"] = "m4a",
  ["audio/m4a"] = "m4a",
  ["audio/x-m4a"] = "m4a",
  ["audio/ogg"] = "ogg",
  ["audio/webm"] = "webm",
  ["audio/flac"] = "flac",
  ["audio/x-flac"] = "flac",
  ["audio/aac"] = "aac",
}

-- Value helpers -----------------------------------------------------------------

local function as_list(value)
  if type(value) == "table" then return value end
  return {}
end

local function non_empty(value)
  if type(value) == "string" and value ~= "" then return value end
  return nil
end

local function non_empty_list(value)
  if type(value) == "table" and #value > 0 then return value end
  return nil
end

-- A list of non-empty strings, or nil when there are none. Never returns an
-- empty table: an empty Lua table has no array form and reaches the router as a
-- JSON object, which fails to unmarshal into the router's []string fields.
local function string_list(value)
  if type(value) ~= "table" then return nil end
  local out = {}
  for _, item in ipairs(value) do
    if type(item) == "string" and item ~= "" then out[#out + 1] = item end
  end
  if #out == 0 then return nil end
  return out
end

local function number_or(value, default)
  if type(value) == "number" then return value end
  return default
end

local function has(list, value)
  for _, item in ipairs(as_list(list)) do
    if item == value then return true end
  end
  return false
end

local function append_unique(list, value)
  if not has(list, value) then list[#list + 1] = value end
  return list
end

local function api_key_of(data)
  if type(data) ~= "table" then return "" end
  return data.api_key or ""
end

local function final_error(message, status, code)
  return { message = message, code = code or "server_error", status = status or 502 }
end

local function message_of(source, fallback)
  if type(source) == "table" and non_empty(source.message) ~= nil then
    return source.message
  end
  return fallback
end

local function transport_error(err)
  return final_error(message_of(err, "upstream request failed"), 502, "server_error")
end

-- Identifier and model-name guards for values placed in URL paths.
local function safe_id(value)
  if type(value) ~= "string" or value == "" or #value > 200 then return nil end
  if string.find(value, "^[%w_%-]+$") == nil then return nil end
  return value
end

local function safe_model_name(name)
  return type(name) == "string" and name ~= "" and #name <= 200
    and string.find(name, "^[%w%._%-%+/:~]+$") ~= nil
    and string.find(name, "..", 1, true) == nil
end

-- TTS voices are provider-specific and the endpoints listing hides them; the
-- speech-filtered model list declares them under supported_voices.
local function probe_voice(model_name, send)
  local resp, call_err = send({
    method = "GET",
    url = BASE_URL .. "/models?output_modalities=speech",
  })
  if call_err ~= nil or resp.status ~= 200 then return nil end
  local ok, page = pcall(json.decode, resp.body or "")
  if not ok or type(page) ~= "table" then return nil end
  for _, model in ipairs(as_list(page.data)) do
    if type(model) == "table" and model.id == model_name then
      return as_list(model.supported_voices)[1]
    end
  end
  return nil
end

-- Tokens an account can afford per OpenRouter's request-size 402:
-- "You requested up to N tokens, but can only afford M."
local function afford_tokens(message)
  if type(message) ~= "string" then return nil end
  return tonumber(string.match(message, "can only afford (%d+)"))
end

-- The provider's own error text under OpenRouter's "Provider returned error"
-- wrapper. The key is usable; the wrapped status belongs to the provider.
local function wrapped_provider_error(envelope, meta)
  if type(envelope) ~= "table" or message_of(envelope) ~= "Provider returned error" then
    return nil
  end
  if type(meta) ~= "table" then return nil end
  return non_empty(meta.raw)
end

-- Headers and timing -------------------------------------------------------------

-- Retry-After as whole seconds. HTTP-date values fall back to the default.
local function retry_after(headers, default)
  local raw = nil
  if type(headers) == "table" then raw = headers["retry-after"] end
  local secs = nil
  if type(raw) == "string" then secs = tonumber(raw) end
  if secs ~= nil and secs > 0 then return math.ceil(secs) end
  return default
end

-- Seconds from now until an ISO 8601 UTC instant such as 2026-10-12T00:00:00.000Z.
-- Days-from-civil arithmetic keeps the result independent of the host time zone.
local function secs_until(iso)
  if type(iso) ~= "string" then return nil end
  local y, mo, d, h, mi, s = string.match(iso, "^(%d+)%-(%d+)%-(%d+)T(%d+):(%d+):(%d+)")
  if y == nil then return nil end
  y, mo, d = tonumber(y), tonumber(mo), tonumber(d)
  h, mi, s = tonumber(h), tonumber(mi), tonumber(s)
  local year = y
  if mo <= 2 then year = year - 1 end
  local era = math.floor(year / 400)
  local yoe = year - era * 400
  local mp = (mo + 9) % 12
  local doy = math.floor((153 * mp + 2) / 5) + d - 1
  local doe = yoe * 365 + math.floor(yoe / 4) - math.floor(yoe / 100) + doy
  local days = era * 146097 + doe - 719468
  return days * 86400 + h * 3600 + mi * 60 + s - os.time()
end

-- Park length for a spent budget window: until its reset when the window has one.
local function budget_wait(resets_at)
  local secs = secs_until(resets_at)
  if secs == nil then return LIFETIME_BUDGET_SECS end
  if secs < 60 then return 60 end
  if secs > MAX_PARK_SECS then return MAX_PARK_SECS end
  return math.ceil(secs)
end

-- The `error` object of an OpenRouter body, or nil. A provider failure after
-- the headers are committed arrives inside the body, so HTTP 200 bodies are
-- checked too: a top-level error, or an error on the first choice.
local function error_envelope(body)
  if type(body) ~= "string" or string.find(body, '"error"', 1, true) == nil then
    return nil
  end
  local ok, parsed = pcall(json.decode, body)
  if not ok or type(parsed) ~= "table" then return nil end
  if type(parsed.error) == "table" then return parsed.error end
  for _, choice in ipairs(as_list(parsed.choices)) do
    if type(choice) == "table" and type(choice.error) == "table" then
      return choice.error
    end
  end
  return nil
end

-- Credential actions -------------------------------------------------------------

-- Rejected key: shared disable when provider automation is on, cooldown park otherwise.
local function bench(ctx, cred_id, reason)
  local cfg = nil
  if type(ctx) == "table" then cfg = ctx.provider_config end
  if type(cfg) == "table" and cfg.disable_failed_credentials == true then
    llm_router.credentials.disable(cred_id, reason)
  else
    llm_router.credentials.park(cred_id, BENCH_SECS, reason)
  end
end

-- Maps one failed attempt to a terminal error plus the next step:
--   "final"      return the error to the client now
--   "credential" stop using this key (it was benched or parked)
--   "skip"       this key cannot handle this attempt; it stays usable
--   "exit"       try the next proxy exit
local function classify_failure(ctx, cred, status, envelope, headers)
  local meta = {}
  if type(envelope) == "table" and type(envelope.metadata) == "table" then
    meta = envelope.metadata
  end

  -- OpenRouter wraps provider-side failures in "Provider returned error": the
  -- key is fine and the wrapped status belongs to the provider, so no bench
  -- and no park on it. Surface the provider's own error text.
  local raw = wrapped_provider_error(envelope, meta)
  if raw ~= nil then
    local st = status
    if type(st) ~= "number" or st < 400 or st > 599 then st = 502 end
    local attributes = { status = st, code = "server_error", cred_id = cred.id }
    local provider_name = non_empty(meta.provider_name)
    if provider_name ~= nil then attributes.detail = provider_name end
    llm_router.log("warn", "provider failed", attributes)
    return final_error(raw, st, "server_error"), "exit"
  end

  if status == 401 then
    bench(ctx, cred.id, "openrouter rejected the api key")
    llm_router.log("warn", "api key rejected", { status = 401, code = "authentication_error", cred_id = cred.id })
    return final_error(message_of(envelope, "openrouter rejected the api key"), 401, "authentication_error"), "credential"
  end

  if status == 402 then
    -- A request larger than the whole in-flight budget will never fit; retrying is pointless.
    if meta.reason == "weight_exceeds_budget" then
      llm_router.log("error", "request exceeds in-flight budget", { status = 402, code = "payment_required", cred_id = cred.id })
      return final_error(message_of(envelope, "request exceeds the in-flight budget"), 402, "payment_required"), "final"
    end
    -- "requires more credits, or fewer max_tokens": the account can still serve
    -- smaller requests, so nothing is exhausted and the caller may clamp the
    -- upstream max_tokens default. Next credential, no park.
    local affordable = afford_tokens(message_of(envelope))
    if affordable ~= nil then
      llm_router.log("warn", "credential cannot afford the request size", { status = 402, code = "payment_required", cred_id = cred.id, detail = affordable })
      local failure = final_error(message_of(envelope, "openrouter credits exhausted"), 402, "payment_required")
      failure.affordable_tokens = affordable
      return failure, "skip"
    end
    -- In-flight holds carry Retry-After; a plain balance shortfall parks for the default.
    local wait = retry_after(headers, PARK_SECS)
    llm_router.credentials.park(cred.id, wait, "openrouter credits exhausted")
    llm_router.log("warn", "credits exhausted", { status = 402, code = "payment_required", cred_id = cred.id, wait = wait })
    return final_error(message_of(envelope, "openrouter credits exhausted"), 402, "payment_required"), "credential"
  end

  if status == 403 then
    -- Budget and key-limit blocks name a limit_source. Guardrail content blocks do
    -- not, and must never be retried on another key: that would bypass the guardrail.
    if meta.limit_source ~= nil then
      local wait = budget_wait(meta.resets_at)
      llm_router.credentials.park(cred.id, wait, "openrouter budget exhausted")
      llm_router.log("warn", "budget limit reached", { status = 403, code = "insufficient_quota", cred_id = cred.id, wait = wait })
      return final_error(message_of(envelope, "openrouter budget exhausted"), 403, "insufficient_quota"), "credential"
    end
    llm_router.log("error", "request forbidden", { status = 403, code = "invalid_request_error", cred_id = cred.id })
    return final_error(message_of(envelope, "openrouter forbade the request"), 403, "invalid_request_error"), "final"
  end

  if status == 429 then
    local wait = retry_after(headers, RATE_LIMIT_SECS)
    llm_router.credentials.park(cred.id, wait, "openrouter rate limited")
    llm_router.log("warn", "rate limited", { status = 429, code = "rate_limit", cred_id = cred.id, wait = wait })
    return final_error(message_of(envelope, "openrouter rate limited"), 429, "rate_limit"), "credential"
  end

  if status == 400 or status == 404 or status == 413 or status == 422 then
    llm_router.log("error", "request rejected", { status = status, code = "invalid_request_error", cred_id = cred.id })
    return final_error(message_of(envelope, "openrouter rejected the request"), status, "invalid_request_error"), "final"
  end

  local st = status
  if type(st) ~= "number" or st < 400 or st > 599 then st = 502 end
  llm_router.log("debug", "unexpected status", { status = st, code = "server_error", cred_id = cred.id, event = "http_error" })
  return final_error(message_of(envelope, "openrouter returned status " .. tostring(st)), st, "server_error"), "exit"
end

-- Proxy exits and payloads -------------------------------------------------------

-- Exits for one attempt. An unset pool goes direct. The pool searches for a live
-- exit when none is ready; a direct pool, the wait deadline or a cancelled
-- request degrade to one direct attempt.
local function pick_proxies(ctx, limit)
  local pool = nil
  local cfg = nil
  if type(ctx) == "table" then cfg = ctx.provider_config end
  if type(cfg) == "table" and type(cfg.proxy) == "table" then
    pool = non_empty(cfg.proxy.pool)
  end
  local res = llm_router.proxies.require({ pool = pool, limit = limit or 3, fallback = "direct" })
  if not res or #as_list(res.proxies) == 0 then
    return { {} }
  end
  return res.proxies
end

-- One request through the selected pool, for every call that is not
-- generation. A 403 (egress block), a 5xx or a transport fault on one exit
-- moves on to the next; a direct pool yields one direct attempt.
local function pooled_request(ctx, client, request)
  local resp, call_err
  for _, px in ipairs(pick_proxies(ctx, 3)) do
    request.proxy_url = px.url
    resp, call_err = client:request(request)
    if call_err == nil and resp.status ~= 403 and resp.status < 500 then break end
  end
  return resp, call_err
end

-- Upstream chat body. Router-only fields never leave the plugin. OpenRouter's
-- own fields (reasoning, plugins, provider, models, usage) pass through as-is.
local function build_payload(request, stream)
  local payload = {}
  for k, v in pairs(request) do
    if k ~= "model_name" and k ~= "model" and k ~= "cache_key" then
      payload[k] = v
    end
  end
  payload.model = request.model_name
  payload.stream = stream
  if not stream then payload.stream_options = nil end
  return payload
end

-- One unary POST across credentials and proxy exits. spec.read turns a success
-- body into the result, or returns nil plus a terminal error for an unusable one.
local function unary_post(ctx, spec)
  local client = llm_router.http_client({ timeout_ms = spec.timeout_ms or UNARY_TIMEOUT_MS })
  local last_err = nil
  local affordable = 0
  for _, cred in ipairs(llm_router.credentials.list()) do
    llm_router.log("debug", "trying credential", { cred_id = cred.id, model = spec.model, endpoint = spec.endpoint })
    local key = api_key_of(cred.data)
    for _, px in ipairs(pick_proxies(ctx, 3)) do
      local resp, call_err = client:request({
        method = "POST",
        url = BASE_URL .. spec.path,
        headers = {
          ["Authorization"] = "Bearer " .. key,
          ["Content-Type"] = spec.content_type or "application/json",
        },
        body = spec.body,
        proxy_url = px.url,
      })
      if call_err ~= nil then
        last_err = transport_error(call_err)
        llm_router.log("debug", "transport error", { cred_id = cred.id, endpoint = spec.endpoint, event = "http_error" })
      else
        local envelope = error_envelope(resp.body)
        local status = resp.status
        if envelope == nil and spec.success[status] then
          local value, read_err = spec.read(resp)
          if value ~= nil then return value, nil end
          return nil, read_err
        end
        if envelope ~= nil and spec.success[status] then
          status = number_or(envelope.code, 502)
        end
        local failure, disposition = classify_failure(ctx, cred, status, envelope, resp.headers)
        if failure.affordable_tokens ~= nil and failure.affordable_tokens > affordable then
          affordable = failure.affordable_tokens
        end
        if disposition == "final" then return nil, failure end
        last_err = failure
        if disposition == "skip" or disposition == "credential" then break end
      end
    end
  end
  local exhausted = last_err or final_error("all openrouter credentials exhausted", 502)
  if affordable > 0 then exhausted.affordable_tokens = affordable end
  llm_router.log("error", "all credentials exhausted", { code = exhausted.code, model = spec.model, reason = exhausted.message })
  return nil, exhausted
end

-- Media helpers -------------------------------------------------------------------

local function audio_format(file_name, content_type)
  if type(file_name) == "string" then
    local ext = string.match(file_name, "%.(%w+)$")
    if ext ~= nil and #ext <= 16 then return string.lower(ext) end
  end
  local mapped = nil
  if type(content_type) == "string" then
    mapped = AUDIO_TYPES[string.lower(content_type)]
  end
  return mapped or "wav"
end

local function image_mime(file_name)
  local ext = ""
  if type(file_name) == "string" then
    ext = string.lower(string.match(file_name, "%.(%w+)$") or "")
  end
  if ext == "jpg" or ext == "jpeg" then return "image/jpeg" end
  if ext == "webp" then return "image/webp" end
  return "image/png"
end

-- Discovery and probes -----------------------------------------------------------

local function endpoint_names_of(paths)
  local names = {}
  for _, path in ipairs(as_list(paths)) do
    local name = ENDPOINT_NAMES[path]
    if name ~= nil then append_unique(names, name) end
  end
  return names
end

-- Catalog rows from GET /models. Rows keep only capabilities this plugin serves.
local function build_model_infos(models)
  local infos = {}
  for _, m in ipairs(models) do
    if type(m) == "table" and non_empty(m.id) ~= nil then
      local endpoints = endpoint_names_of(m.endpoints)
      if #endpoints > 0 then
        local arch = as_list(m.architecture)
        local top = as_list(m.top_provider)
        local row = {
          name = m.id,
          display_name = non_empty(m.name) or m.id,
          endpoints = endpoints,
          context_window = number_or(m.context_length, 0),
          max_tokens = number_or(top.max_completion_tokens, 0),
          -- Omitted when empty: a missing field is a nil slice on the router side.
          input_modalities = string_list(arch.input_modalities),
          output_modalities = string_list(arch.output_modalities),
          supported_parameters = string_list(m.supported_parameters),
          supported_voices = string_list(m.supported_voices),
        }
        local reasoning = as_list(m.reasoning)
        local efforts = string_list(reasoning.supported_efforts)
        if efforts ~= nil then
          row.reasoning = {
            supported_efforts = efforts,
            default_effort = non_empty(reasoning.default_effort),
            default_enabled = reasoning.default_enabled,
          }
        end
        infos[#infos + 1] = row
      end
    end
  end
  table.sort(infos, function(a, b) return a.name < b.name end)
  return infos
end

-- Whether the model advertises any output modality the endpoint needs.
local function supports_endpoint(data, wanted)
  local outputs = as_list(as_list(data.architecture).output_modalities)
  for _, modality in ipairs(wanted) do
    if has(outputs, modality) then return true end
  end
  return false
end

-- Probe models -------------------------------------------------------------------

-- One transparent 1x1 pixel for vision probes.
local PROBE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAEklEQVR4nAAFAPr/AAAAAAADAAAFAAH5Uqg1AAAAAElFTkSuQmCC"

-- One second of silence as an 8 kHz 8-bit mono wav, for transcription probes.
local function probe_wav()
  local silence = string.rep("\0", 8000)
  local header = "RIFF" .. string.char(0x64, 0x1f, 0x00, 0x00) .. "WAVE"
    .. "fmt " .. string.char(0x10, 0x00, 0x00, 0x00)
    .. string.char(0x01, 0x00) .. string.char(0x01, 0x00)
    .. string.char(0x40, 0x1f, 0x00, 0x00) .. string.char(0x40, 0x1f, 0x00, 0x00)
    .. string.char(0x01, 0x00) .. string.char(0x08, 0x00)
    .. "data" .. string.char(0x40, 0x1f, 0x00, 0x00)
  return llm_router.base64_encode(header .. silence)
end

-- Minimal probe request for one endpoint: {path, accept, body, read}. The read
-- function turns a usable response into a short result or a terminal error.
local function probe_request(endpoint, model_name, voice)
  if endpoint == "chat" or endpoint == "chat/completions" then
    return {
      path = "/chat/completions",
      accept = { [200] = true },
      body = json.encode({
        model = model_name,
        messages = { { role = "user", content = "Reply with the single word: ok" } },
        max_tokens = 8,
      }),
      read = function() return { response = "ok" } end,
    }
  end
  if endpoint == "vision" then
    return {
      path = "/chat/completions",
      accept = { [200] = true },
      body = json.encode({
        model = model_name,
        messages = { { role = "user", content = {
          { type = "text", text = "Reply with the single word: pixel" },
          { type = "image_url", image_url = { url = "data:image/png;base64," .. PROBE_PNG } },
        } } },
        max_tokens = 8,
      }),
      read = function() return { response = "ok" } end,
    }
  end
  if endpoint == "embeddings" then
    return {
      path = "/embeddings",
      accept = { [200] = true },
      body = json.encode({ model = model_name, input = { "availability probe" } }),
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if ok and type(out) == "table" and type(out.data) == "table" and #out.data > 0 then
          return { response = "ok" }
        end
        return nil, final_error("openrouter returned invalid embeddings", 502)
      end,
    }
  end
  if endpoint == "speech" or endpoint == "audio/speech" then
    -- Upstream format defaults differ per provider: HeyGen serves pcm only,
    -- MiniMax unary serves mp3 only. The probe walks omit, mp3, pcm.
    local bodies = {}
    local formats = { "mp3", "pcm" }
    for i = 0, #formats do
      local body = { model = model_name, input = "hello" }
      if voice ~= nil then body.voice = voice end
      if i > 0 then body.response_format = formats[i] end
      bodies[#bodies + 1] = json.encode(body)
    end
    return {
      path = "/audio/speech",
      accept = { [200] = true },
      body = bodies[1],
      alt_bodies = { bodies[2], bodies[3] },
      read = function(resp)
        if type(resp.body) == "string" and resp.body ~= "" then
          return { response = "ok" }
        end
        return nil, final_error("openrouter returned empty audio", 502)
      end,
    }
  end
  if endpoint == "transcription" or endpoint == "audio/transcriptions" then
    return {
      path = "/audio/transcriptions",
      accept = { [200] = true },
      body = json.encode({ model = model_name, input_audio = { data = probe_wav(), format = "wav" } }),
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if ok and type(out) == "table" and out.text ~= nil then
          return { response = "ok" }
        end
        return nil, final_error("openrouter returned an invalid transcription", 502)
      end,
    }
  end
  if endpoint == "image" or endpoint == "images/generations" then
    return {
      path = "/images",
      accept = { [200] = true },
      body = json.encode({ model = model_name, prompt = "a small dot" }),
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if ok and type(out) == "table" and type(out.data) == "table" and #out.data > 0 then
          return { response = "ok" }
        end
        return nil, final_error("openrouter returned no images", 502)
      end,
    }
  end
  if endpoint == "video" or endpoint == "videos" then
    return {
      path = "/videos",
      accept = { [200] = true, [202] = true },
      body = json.encode({ model = model_name, prompt = "a small dot" }),
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if ok and type(out) == "table" and non_empty(out.id) ~= nil then
          return { response = "ok" }
        end
        return nil, final_error("openrouter returned no video job", 502)
      end,
    }
  end
  return nil
end

-- One probe generation attempt for one credential, through the selected pool,
-- and no park or bench on any outcome. Returns {response} or nil with a
-- classified failure {kind, message}.
local function probe_attempt(ctx, spec, cred)
  local client = llm_router.http_client({ timeout_ms = PROBE_GENERATION_TIMEOUT_MS })
  local bodies = { spec.body }
  for _, alt in ipairs(as_list(spec.alt_bodies)) do
    bodies[#bodies + 1] = alt
  end
  local failure = nil
  for index, body in ipairs(bodies) do
    local resp, call_err = pooled_request(ctx, client, {
      method = "POST",
      url = BASE_URL .. spec.path,
      headers = {
        ["Authorization"] = "Bearer " .. api_key_of(cred.data),
        ["Content-Type"] = "application/json",
      },
      body = body,
    })
    if call_err ~= nil then
      return nil, { kind = "transport", message = message_of(call_err, "upstream request failed") }
    end
    local status = resp.status
    local envelope = error_envelope(resp.body)
    if envelope ~= nil and spec.accept[status] == true then
      status = number_or(envelope.code, 502)
    end
    if spec.accept[status] == true then
      return spec.read(resp)
    end
    local message = message_of(envelope, "request failed with status " .. tostring(status))
    local raw = wrapped_provider_error(envelope, type(envelope) == "table" and envelope.metadata)
    local kind = "other"
    if raw ~= nil then
      kind, message = "provider", raw
    elseif status == 429 then
      kind = "quota"
    elseif status == 402 then
      kind = "payment"
    elseif status == 401 then
      kind = "rejected"
    elseif status == 404 then
      kind = "missing"
    elseif type(status) == "number" and status >= 500 then
      kind = "provider"
    end
    failure = { kind = kind, message = message }
    if failure.kind ~= "other" or index == #bodies then break end
  end
  return nil, failure
end

-- GET for a video job, tried across credentials: the job belongs to the
-- account that created it, and a key from another account sees a 404.
local function job_request(ctx, path)
  local creds = llm_router.credentials.list()
  if #creds == 0 then
    return nil, final_error("no openrouter credentials configured", 400, "invalid_request_error")
  end
  local client = llm_router.http_client({ timeout_ms = UNARY_TIMEOUT_MS })
  local last_err = nil
  for _, cred in ipairs(creds) do
    local resp, call_err = pooled_request(ctx, client, {
      method = "GET",
      url = BASE_URL .. path,
      headers = { ["Authorization"] = "Bearer " .. api_key_of(cred.data) },
    })
    if call_err ~= nil then
      last_err = transport_error(call_err)
    elseif resp.status == 200 then
      return resp, nil
    elseif resp.status == 404 then
      last_err = final_error("video job not found", 404, "invalid_request_error")
    else
      local failure, disposition = classify_failure(ctx, cred, resp.status, error_envelope(resp.body), resp.headers)
      if disposition == "final" then return nil, failure end
      last_err = failure
    end
  end
  return nil, last_err or final_error("all openrouter credentials exhausted", 502)
end

-- One streaming chat attempt across credentials and proxy exits. Defined before
-- the register table because complete_stream retries it with clamped max_tokens.
local function stream_once(ctx, payload, request, emit)
  local client = llm_router.http_client({ timeout_ms = STREAM_TIMEOUT_MS })
  local last_err = nil
  local affordable = 0
  for _, cred in ipairs(llm_router.credentials.list()) do
    llm_router.log("debug", "trying credential", { cred_id = cred.id, model = request.model })
    local key = api_key_of(cred.data)
    for _, px in ipairs(pick_proxies(ctx, 3)) do
      local emitted = false
      local failure = nil
      local disposition = nil
      local mid_error = nil
      local _, stream_err = client:stream({
        method = "POST",
        url = BASE_URL .. "/chat/completions",
        headers = { ["Authorization"] = "Bearer " .. key, ["Content-Type"] = "application/json" },
        body = json.encode(payload),
        proxy_url = px.url,
        on_response = function(r)
          if r.status == 200 then return nil end
          failure, disposition = classify_failure(ctx, cred, r.status, error_envelope(r.body), r.headers)
          return failure
        end,
        on_line = function(line)
          -- SSE comments such as ": OPENROUTER PROCESSING" are skipped by this branch.
          local data = nil
          if string.sub(line, 1, 6) == "data: " then
            data = string.sub(line, 7)
          elseif string.sub(line, 1, 1) == "{" then
            data = line
          else
            return
          end
          if data == "[DONE]" then return end
          local ok, chunk = pcall(json.decode, data)
          if not ok or type(chunk) ~= "table" then return end
          local err_obj = chunk.error
          if type(err_obj) ~= "table" then
            local first = as_list(chunk.choices)[1]
            if type(first) == "table" then err_obj = first.error end
          end
          if type(err_obj) == "table" then
            mid_error = err_obj
            return
          end
          -- Emitted chunks need choices or usage. The usage chunk carries one choice.
          local has_choices = type(chunk.choices) == "table" and #chunk.choices > 0
          if not has_choices and chunk.usage == nil then return end
          chunk.model = request.model
          emitted = true
          emit(chunk)
        end,
      })
      if stream_err ~= nil then
        if failure == nil then
          failure = transport_error(stream_err)
          disposition = "exit"
        end
      elseif mid_error ~= nil then
        failure, disposition = classify_failure(ctx, cred, number_or(mid_error.code, 502), mid_error, nil)
        llm_router.log("warn", "stream failed", { code = failure.code, cred_id = cred.id, event = "stream_error" })
      else
        return nil, nil
      end
      if emitted then return nil, failure end
      if failure.affordable_tokens ~= nil and failure.affordable_tokens > affordable then
        affordable = failure.affordable_tokens
      end
      last_err = failure
      if disposition == "final" then return nil, failure end
      if disposition == "skip" or disposition == "credential" then break end
    end
  end
  local exhausted = last_err or final_error("all openrouter credentials exhausted", 502)
  if affordable > 0 then exhausted.affordable_tokens = affordable end
  llm_router.log("error", "all credentials exhausted", { code = exhausted.code, model = request.model, reason = exhausted.message })
  return nil, exhausted
end

-- Registration -------------------------------------------------------------------

llm_router.register("openrouter", {
  icon = "https://openrouter.ai/favicon.ico",

  proxy_schema = {},

  locales = {
    ["en"] = {
      ["openrouter.subtitle"] = "Get your API key from the OpenRouter keys page.",
      ["openrouter.open_keys"] = "Open OpenRouter Keys",
      ["misc.api_key"] = "API Key",
      ["misc.save"] = "Save",
    },
    ["ru"] = {
      ["openrouter.subtitle"] = "Получите API-ключ на странице ключей OpenRouter.",
      ["openrouter.open_keys"] = "Открыть ключи OpenRouter",
      ["misc.api_key"] = "API-ключ",
      ["misc.save"] = "Сохранить",
    },
  },

  credential_schema = {
    {
      type = "section",
      title = "OpenRouter",
      subtitle = "openrouter.subtitle",
      content = {
        { type = "link", text = "openrouter.open_keys", url = "https://openrouter.ai/settings/keys" },
        { type = "secret", name = "api_key", label = "misc.api_key", required = true },
        { type = "button", text = "misc.save", form_action = "submit" },
      },
    },
  },

  validate_credentials = function(data)
    local key = nil
    if type(data) == "table" then key = data.api_key end
    if type(key) ~= "string" or key == "" then
      return false, { message = "api_key is required", code = "invalid_request_error" }
    end
    if #key < 20 or string.find(key, "%s") ~= nil then
      return false, { message = "api_key must be at least 20 characters without whitespace", code = "invalid_request_error" }
    end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, final_error("no openrouter credentials configured", 400, "invalid_request_error")
    end
    -- Discovery needs a single working key. output_modalities=all returns every
    -- model; a missing parameter would return text models only.
    local client = llm_router.http_client({ timeout_ms = DISCOVERY_TIMEOUT_MS })
    local last_err = nil
    for _, cred in ipairs(creds) do
      local resp, call_err = pooled_request(ctx, client, {
        method = "GET",
        url = BASE_URL .. "/models?output_modalities=all",
        headers = { ["Authorization"] = "Bearer " .. api_key_of(cred.data) },
      })
      if call_err ~= nil then
        last_err = transport_error(call_err)
      elseif resp.status == 200 then
        local ok, page = pcall(json.decode, resp.body or "")
        if not ok or type(page) ~= "table" or type(page.data) ~= "table" then
          return nil, final_error("openrouter model discovery returned invalid JSON", 502)
        end
        local infos = build_model_infos(page.data)
        if #infos == 0 then
          return nil, final_error("openrouter lists no models this plugin can serve", 502, "server_error")
        end
        return infos, nil
      else
        local st = resp.status
        if st < 400 or st > 599 then st = 502 end
        last_err = final_error(
          message_of(error_envelope(resp.body), "openrouter model discovery failed with status " .. tostring(resp.status)),
          st,
          "server_error"
        )
      end
    end
    return nil, last_err or final_error("all openrouter credentials exhausted", 502)
  end,

  -- Key liveness from GET /key. A rejected key is unhealthy. An exhausted budget
  -- is not dead (balances can be topped up), so it reports unknown and changes nothing.
  check_health = function(ctx, credential)
    local key = api_key_of(credential.data)
    if key == "" then return { status = "unhealthy", message = "api_key is required" } end
    local client = llm_router.http_client({ timeout_ms = PROBE_TIMEOUT_MS })
    local resp, call_err = pooled_request(ctx, client, {
      method = "GET",
      url = BASE_URL .. "/key",
      headers = { ["Authorization"] = "Bearer " .. key },
    })
    if call_err ~= nil then
      return { status = "unknown", message = message_of(call_err, "request failed") }
    end
    if resp.status == 401 then return { status = "unhealthy", message = "api key rejected" } end
    if resp.status ~= 200 then return { status = "unknown", message = "status " .. tostring(resp.status) } end
    local ok, page = pcall(json.decode, resp.body or "")
    if not ok or type(page) ~= "table" or type(page.data) ~= "table" then
      return { status = "unknown", message = "invalid key information" }
    end
    local info = page.data
    if info.is_management_key == true then
      return { status = "unhealthy", message = "management keys cannot run inference" }
    end
    if type(info.limit_remaining) == "number" and info.limit_remaining <= 0 then
      return { status = "unknown", message = "key credit limit reached" }
    end
    local budget = as_list(info.effective_budget)
    if type(budget.remaining_usd) == "number" and budget.remaining_usd <= 0 then
      return { status = "unknown", message = "budget limit reached" }
    end
    return { status = "healthy" }
  end,

  -- Model liveness as real availability: the endpoints listing rules the model
  -- out when it is missing or cannot serve the endpoint, then a minimal
  -- generation runs on every available account until one succeeds. Explicit
  -- dashboard probes only; probes never park or bench, and 402s are account
  -- state, not model death.
  check_model = function(ctx, model_ref)
    local endpoint = model_ref.endpoint or "chat"
    local wanted = CAPABILITY_OUTPUT[endpoint]
    if wanted == nil then
      return { status = "unknown", message = "endpoint not supported: " .. tostring(endpoint) }
    end
    local model_name = model_ref.model_name or ""
    if model_name == "" then return { status = "unknown", message = "model name is required" } end
    if not safe_model_name(model_name) then return { status = "unknown", message = "invalid model name" } end

    local creds = llm_router.credentials.list()
    if #creds == 0 then return { status = "unknown", message = "no credentials available" } end

    -- Phase 1: the endpoints listing rules out dead models cheaply. It never
    -- proves availability on its own.
    local client = llm_router.http_client({ timeout_ms = PROBE_TIMEOUT_MS })
    local tried = 0
    local missing = 0
    local seen_unknown = nil
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key == "" then
        seen_unknown = seen_unknown or "api_key is required"
      else
        tried = tried + 1
        llm_router.log("debug", "probing model", { cred_id = cred.id, model = model_ref.model, endpoint = endpoint })
        local resp, call_err = pooled_request(ctx, client, {
          method = "GET",
          url = BASE_URL .. "/models/" .. model_name .. "/endpoints",
          headers = { ["Authorization"] = "Bearer " .. key },
        })
        if call_err ~= nil then
          seen_unknown = seen_unknown or "probe request failed"
        elseif resp.status == 200 then
          local ok, page = pcall(json.decode, resp.body or "")
          if not ok or type(page) ~= "table" or type(page.data) ~= "table" then
            seen_unknown = seen_unknown or "invalid model listing"
          elseif not supports_endpoint(page.data, wanted) then
            return { status = "unknown", message = "endpoint not supported: " .. tostring(endpoint) }
          end
        elseif resp.status == 404 then
          missing = missing + 1
        else
          seen_unknown = seen_unknown or ("status " .. tostring(resp.status))
        end
      end
    end
    if tried > 0 and missing == tried then
      return { status = "unhealthy", message = "model not found upstream" }
    end

    -- Phase 2: a minimal generation per available account, first success wins.
    -- Speech resolves a real supported voice first; without it most providers
    -- reject the probe on missing voice, not on model state.
    local voice = nil
    if endpoint == "speech" or endpoint == "audio/speech" then
      voice = probe_voice(model_name, function(request) return pooled_request(ctx, client, request) end)
    end
    local spec = probe_request(endpoint, model_name, voice)
    local quota_seen = false
    local provider_seen = false
    local missing_seen = false
    local provider_message = nil
    local payments = 0
    for _, cred in ipairs(creds) do
      local key = api_key_of(cred.data)
      if key ~= "" then
        llm_router.log("debug", "probing model generation", { cred_id = cred.id, model = model_ref.model, endpoint = endpoint })
        local value, failure = probe_attempt(ctx, spec, cred)
        if value ~= nil then
          return { status = "healthy", response = value.response or "ok" }
        end
        if failure == nil then
          seen_unknown = seen_unknown or "probe failed"
        elseif failure.kind == "quota" then
          quota_seen = true
        elseif failure.kind == "payment" then
          payments = payments + 1
        elseif failure.kind == "provider" then
          provider_seen = true
          provider_message = failure.message
        elseif failure.kind == "missing" then
          missing_seen = true
        else
          seen_unknown = seen_unknown or failure.message
        end
      end
    end
    -- Quota outranks provider failures: a rate limit is transient per account,
    -- an upstream provider outage is model state. Nothing paid for means the
    -- model is alive but no account can afford it.
    if quota_seen then
      return { status = "quota_exceeded", message = "quota exhausted" }
    end
    if missing_seen then
      return { status = "unhealthy", message = "model not found upstream" }
    end
    if provider_seen then
      return { status = "unhealthy", message = provider_message or "provider failed upstream" }
    end
    if tried > 0 and payments == tried then
      return { status = "quota_exceeded", message = "no credential can afford this model" }
    end
    return { status = "unknown", message = seen_unknown or "health check inconclusive" }
  end,

  -- Chat completions. OpenRouter returns OpenAI-shaped results and normalizes the
  -- model's own finish reasons, so the body passes through with the router's model id.
  complete = function(ctx, request)
    local payload = build_payload(request, false)
    local spec = {
      model = request.model,
      endpoint = "chat/completions",
      path = "/chat/completions",
      body = json.encode(payload),
      success = { [200] = true },
      timeout_ms = UNARY_TIMEOUT_MS,
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if not ok or type(out) ~= "table" then
          return nil, final_error("openrouter returned invalid JSON", 502)
        end
        if type(out.choices) ~= "table" or #out.choices == 0 then
          return nil, final_error("openrouter returned no choices", 502)
        end
        out.model = request.model
        return out
      end,
    }
    local result, err = unary_post(ctx, spec)
    -- The afford 402 lands on the upstream max_tokens default, not on the
    -- request: retry once at the affordable token count.
    if err ~= nil and err.affordable_tokens ~= nil and payload.max_tokens == nil then
      llm_router.log("debug", "clamped max_tokens to the affordable limit", { model = request.model, detail = err.affordable_tokens })
      payload.max_tokens = err.affordable_tokens
      spec.body = json.encode(payload)
      result, err = unary_post(ctx, spec)
    end
    return result, err
  end,

  -- Streaming chat. Chunks are emitted as they arrive. A failure before the first
  -- emit fails over (next proxy exit, next key). A failure after it is final,
  -- because the client already holds part of the answer. An afford 402 on the
  -- upstream max_tokens default retries once with the affordable token count.
  complete_stream = function(ctx, request, emit)
    local payload = build_payload(request, true)
    local result, err = stream_once(ctx, payload, request, emit)
    if err ~= nil and err.affordable_tokens ~= nil and payload.max_tokens == nil then
      llm_router.log("debug", "clamped max_tokens to the affordable limit", { model = request.model, detail = err.affordable_tokens })
      payload.max_tokens = err.affordable_tokens
      result, err = stream_once(ctx, payload, request, emit)
    end
    return result, err
  end,

  embed = function(ctx, request)
    local input = request.input
    if type(input) == "string" then input = { input } end
    if type(input) ~= "table" or #input == 0 then
      return nil, final_error("input is required", 400, "invalid_request_error")
    end
    return unary_post(ctx, {
      model = request.model,
      endpoint = "embeddings",
      path = "/embeddings",
      body = json.encode({
        model = request.model_name,
        input = input,
        encoding_format = non_empty(request.encoding_format),
        dimensions = request.dimensions,
      }),
      success = { [200] = true },
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if not ok or type(out) ~= "table" or type(out.data) ~= "table" then
          return nil, final_error("openrouter returned invalid embeddings", 502)
        end
        if #out.data ~= #input then
          return nil, final_error("openrouter returned a different number of embeddings", 502)
        end
        out.model = request.model
        return out
      end,
    })
  end,

  -- Speech-to-text as JSON with base64 audio (OpenRouter's documented inline form).
  -- Verbose output is requested only when segments or timestamps are asked for.
  -- OpenRouter has no prompt field, so a prompt is dropped with a debug line.
  transcribe = function(ctx, request)
    local task = request.task or "transcribe"
    if task == "translate" then
      return nil, final_error("openrouter has no audio translation endpoint", 400, "invalid_request_error")
    end
    if task ~= "transcribe" then
      return nil, final_error("unsupported audio task: " .. tostring(task), 400, "invalid_request_error")
    end
    if type(request.file) ~= "string" or request.file == "" then
      return nil, final_error("audio file is required", 400, "invalid_request_error")
    end
    if non_empty(request.prompt) ~= nil then
      llm_router.log("debug", "transcription prompt is not supported by openrouter", { model = request.model })
    end

    local granularities = {}
    for _, g in ipairs(as_list(request.timestamp_granularities)) do
      if g == "word" or g == "segment" then append_unique(granularities, g) end
    end
    if request.needs_segments == true then append_unique(granularities, "segment") end
    local verbose = request.needs_segments == true
      or request.response_format == "verbose_json"
      or #granularities > 0
    if verbose and #granularities == 0 then granularities[1] = "segment" end

    return unary_post(ctx, {
      model = request.model,
      endpoint = "audio/transcriptions",
      path = "/audio/transcriptions",
      body = json.encode({
        model = request.model_name,
        input_audio = {
          data = llm_router.base64_encode(request.file),
          format = audio_format(request.file_name, request.content_type),
        },
        language = non_empty(request.language),
        temperature = request.temperature,
        response_format = verbose and "verbose_json" or nil,
        timestamp_granularities = verbose and granularities or nil,
      }),
      success = { [200] = true },
      timeout_ms = MEDIA_TIMEOUT_MS,
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if not ok or type(out) ~= "table" or type(out.text) ~= "string" then
          return nil, final_error("openrouter returned an invalid transcription", 502)
        end
        return out
      end,
    })
  end,

  -- Text-to-speech. OpenRouter returns raw audio: mp3 or 16-bit little-endian pcm.
  -- mp3 is the default because it plays everywhere.
  speech = function(ctx, request)
    if non_empty(request.input) == nil then
      return nil, final_error("input is required", 400, "invalid_request_error")
    end
    local format = non_empty(request.response_format) or "mp3"
    if format ~= "mp3" and format ~= "pcm" then
      return nil, final_error("openrouter speech supports response_format mp3 or pcm", 400, "invalid_request_error")
    end
    return unary_post(ctx, {
      model = request.model,
      endpoint = "audio/speech",
      path = "/audio/speech",
      body = json.encode({
        model = request.model_name,
        input = request.input,
        voice = non_empty(request.voice),
        response_format = format,
        speed = request.speed,
        instructions = non_empty(request.instructions),
      }),
      success = { [200] = true },
      timeout_ms = MEDIA_TIMEOUT_MS,
      read = function(resp)
        if type(resp.body) ~= "string" or resp.body == "" then
          return nil, final_error("openrouter returned empty audio", 502)
        end
        return { audio_b64 = llm_router.base64_encode(resp.body), format = format }
      end,
    })
  end,

  -- Image generation and single-image edits. OpenRouter returns base64 only, so
  -- url responses and masks are refused. Style has no OpenRouter equivalent and is dropped.
  generate_image = function(ctx, request)
    if non_empty(request.prompt) == nil then
      return nil, final_error("prompt is required", 400, "invalid_request_error")
    end
    if non_empty(request.mask_b64) ~= nil then
      return nil, final_error("openrouter image edits do not accept a mask", 400, "invalid_request_error")
    end
    if request.response_format == "url" then
      return nil, final_error("openrouter returns images as base64 only", 400, "invalid_request_error")
    end
    if request.style ~= nil then
      llm_router.log("debug", "image style is not supported by openrouter", { model = request.model })
    end
    local payload = {
      model = request.model_name,
      prompt = request.prompt,
      n = request.n,
      size = non_empty(request.size),
      quality = non_empty(request.quality),
    }
    if non_empty(request.image_b64) ~= nil then
      payload.input_references = {
        {
          type = "image_url",
          image_url = { url = "data:" .. image_mime(request.image_name) .. ";base64," .. request.image_b64 },
        },
      }
    end
    return unary_post(ctx, {
      model = request.model,
      endpoint = "images/generations",
      path = "/images",
      body = json.encode(payload),
      success = { [200] = true },
      timeout_ms = MEDIA_TIMEOUT_MS,
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if not ok or type(out) ~= "table" or type(out.data) ~= "table" or #out.data == 0 then
          return nil, final_error("openrouter returned no images", 502)
        end
        return out
      end,
    })
  end,

  -- Video submit. The job runs asynchronously; poll_video and video_content follow it.
  -- callback_url is router storage only and is not forwarded.
  generate_video = function(ctx, request)
    return unary_post(ctx, {
      model = request.model,
      endpoint = "videos",
      path = "/videos",
      body = json.encode({
        model = request.model_name,
        prompt = non_empty(request.prompt),
        duration = request.duration,
        resolution = non_empty(request.resolution),
        aspect_ratio = non_empty(request.aspect_ratio),
        size = non_empty(request.size),
        seed = request.seed,
        generate_audio = request.generate_audio,
        frame_images = non_empty_list(request.frame_images),
        input_references = non_empty_list(request.input_references),
        previous_job_id = non_empty(request.previous_job_id),
        user = non_empty(request.user),
        session_id = non_empty(request.session_id),
        provider = type(request.provider) == "table" and request.provider or nil,
      }),
      success = { [200] = true, [202] = true },
      timeout_ms = UNARY_TIMEOUT_MS,
      read = function(resp)
        local ok, out = pcall(json.decode, resp.body or "")
        if not ok or type(out) ~= "table" or non_empty(out.id) == nil then
          return nil, final_error("openrouter returned no video job", 502)
        end
        return { id = out.id, status = non_empty(out.status) or "pending", generation_id = out.generation_id }
      end,
    })
  end,

  poll_video = function(ctx, request)
    local job_id = safe_id(request.job_id)
    if job_id == nil then
      return nil, final_error("job_id is invalid", 400, "invalid_request_error")
    end
    local resp, err = job_request(ctx, "/videos/" .. job_id)
    if resp == nil then return nil, err end
    local ok, out = pcall(json.decode, resp.body or "")
    if not ok or type(out) ~= "table" then
      return nil, final_error("openrouter returned invalid JSON", 502)
    end
    return out, nil
  end,

  video_content = function(ctx, request)
    local job_id = safe_id(request.job_id)
    if job_id == nil then
      return nil, final_error("job_id is invalid", 400, "invalid_request_error")
    end
    local index = 0
    if type(request.index) == "number" then index = request.index end
    if index < 0 or index ~= math.floor(index) then
      return nil, final_error("index must be a non-negative integer", 400, "invalid_request_error")
    end
    local resp, err = job_request(ctx, "/videos/" .. job_id .. "/content?index=" .. string.format("%d", math.floor(index)))
    if resp == nil then return nil, err end
    if type(resp.body) ~= "string" or resp.body == "" then
      return nil, final_error("openrouter returned empty video content", 502)
    end
    local content_type = "video/mp4"
    if type(resp.headers) == "table" and non_empty(resp.headers["content-type"]) ~= nil then
      content_type = resp.headers["content-type"]
    end
    return { video_b64 = llm_router.base64_encode(resp.body), content_type = content_type }, nil
  end,
})
