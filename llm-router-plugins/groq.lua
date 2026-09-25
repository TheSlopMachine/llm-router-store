--- @plugin Groq
--- @author TheSlopMachine
--- @version 2.0.1
--- @router_version 0.1.1
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

local function api_key_of(credential)
  if credential == nil or credential.data == nil then return "" end
  return credential.data.api_key or ""
end

local function classify_extension(raw, default_err)
  if raw.status == 403 then
    -- Groq answers 401 for bad keys; 403 ("Forbidden", "Access denied.
    -- Please check your network settings.") is the Cloudflare egress-IP
    -- block: the exit is at fault, not the credential.
    local message = default_err and default_err.message or tostring(raw.body)
    return { type = "geo", message = message }
  end
  if raw.status ~= 429 then return nil end
  local err = default_err or { type = "rate_limit", message = tostring(raw.body) }
  -- Daily-quota exhaustion (RPD) is account-scoped; minute limits stay
  -- account-scoped too: Groq limits bind to the key, never to the IP.
  local lower = string.lower(err.message or "")
  if string.find(lower, "per day", 1, true) then
    err.type = "quota_exceeded"
  end
  err.scope = { "account" }
  return err
end

-- Build the upstream payload from the OpenAI-shaped request table:
-- pass everything through, fix the model id, pin the stream flag.
-- Router-only fields (model_name) never leave the plugin.
local function build_payload(request, stream)
  local payload = {}
  for k, v in pairs(request) do
    if k ~= "model_name" then payload[k] = v end
  end
  payload.model = request.model_name
  payload.stream = stream
  if not stream then
    payload.stream_options = nil
  end
  return payload
end

llm_router.register("groq", {
  icon = "https://console.groq.com/favicon.ico",

  classify_error = classify_extension,

  credential_schema = function()
    return {
      { type = "section", title = "Groq",
        subtitle = "Get your API key from the Groq console.",
        content = {
          { type = "link", text = "Open Groq Console", url = "https://console.groq.com/keys" },
          { type = "secret", name = "api_key", label = "API Key", required = true },
          { type = "button", text = "Save", form_action = "submit" },
        } },
    }
  end,

  validate_credentials = function(data)
    local key = data.api_key
    if key == nil or key == "" then
      return false, { type = "invalid_request", message = "api_key is required" }
    end
    if #key < 20 then
      return false, { type = "invalid_request", message = "api_key: minimum 20 characters" }
    end
    return true
  end,

  get_model_infos = function(ctx, credential, provider_config)
    local client = llm_router.http_client({})
    local resp, err = client:request({
      method = "GET", url = BASE_URL .. "/models",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential) },
    })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, llm_router.classify_error({ status = resp.status, headers = resp.headers, body = resp.body })
    end
    local page = json.decode(resp.body)
    local infos = {}
    for _, m in ipairs(page.data or {}) do
      -- Split by modality: audio-in/text-out models are speech-to-text,
      -- text-in/text-out are chat. Text-in/audio-out (playai-tts) waits for
      -- the speech endpoint.
      local function has(list, v)
        for _, x in ipairs(list or {}) do if x == v then return true end end
        return false
      end
      if m.active and has(m.input_modalities, "audio") and has(m.output_modalities, "text") then
        local limits = RATE_LIMITS[m.id] or { rpm = 20, rpd = 2000, tpm = 0 }
        table.insert(infos, {
          name = m.id,
          display_name = (m.name and m.name ~= "") and m.name or m.id,
          rpm = limits.rpm, tpm = limits.tpm, rpd = limits.rpd,
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
          rpm = limits.rpm, tpm = limits.tpm, rpd = limits.rpd,
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

  complete = function(ctx, credential, request)
    local client = llm_router.http_client({})
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/chat/completions",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(build_payload(request, false)),
    })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, llm_router.classify_error({ status = resp.status, headers = resp.headers, body = resp.body })
    end
    local out = json.decode(resp.body)
    out.model = request.model
    return out
  end,

  complete_stream = function(ctx, credential, request, emit)
    local client = llm_router.http_client({})
    local full_model = request.model
    local resp, stream_err = client:stream({
      method = "POST", url = BASE_URL .. "/chat/completions",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(build_payload(request, true)),
      on_response = function(r)
        if r.status ~= 200 then
          return llm_router.classify_error({ status = r.status, headers = r.headers, body = r.body })
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
      end,
    })
    if stream_err then
      return nil, stream_err
    end
  end,

  -- Speech-to-text: always fetch verbose_json upstream; the router renders
  -- the client's response_format (json/text/srt/vtt) from the normalized
  -- response, so segments must be present when the client asked for them.
  transcribe = function(ctx, credential, request)
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
      for _, x in ipairs(gran) do if x == g then return end end
      gran[#gran + 1] = g
    end
    if request.needs_segments then add_gran("segment") end
    for _, g in ipairs(request.timestamp_granularities or {}) do add_gran(g) end
    for _, g in ipairs(gran) do
      parts[#parts + 1] = { name = "timestamp_granularities[]", value = g }
    end
    local body, ctype = llm_router.multipart(parts)
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/audio/transcriptions",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential), ["Content-Type"] = ctype },
      body = body,
    })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, llm_router.classify_error({ status = resp.status, headers = resp.headers, body = resp.body })
    end
    return json.decode(resp.body)
  end,
})
