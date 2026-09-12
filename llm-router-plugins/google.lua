--- @plugin Google AI Studio
--- @author TheSlopMachine
--- @version 1.2.0
--- @router_version 0.0.4
--- @description Google Gemini models via AI Studio API
--- @allow_host generativelanguage.googleapis.com
--- @proxy_location US
--- @proxy_force_on_mismatch true

local BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

local function classify_error(status, body)
  local message = body
  local ok, parsed = pcall(json.decode, body)
  if ok and parsed and parsed.error and type(parsed.error.message) == "string" and parsed.error.message ~= "" then
    message = parsed.error.message
  end
  if status == 401 or status == 403 then
    return nil, { type = "auth", message = message }
  elseif status == 400 and message:find("location is not supported") then
    -- Geo-blocked: the proxy exit is at fault, not the request.
    return nil, { type = "geo", message = message }
  elseif status == 429 then
    return nil, { type = "rate_limit", message = message, retry_after = os.time() + 60 }
  elseif status == 408 or status == 504 then
    return nil, { type = "timeout", message = message }
  elseif status >= 500 then
    return nil, { type = "upstream", message = message }
  end
  return nil, { type = "invalid_request", message = message }
end

local function api_key_of(credential)
  if credential == nil or credential.data == nil then return "" end
  return credential.data.api_key or ""
end

local function map_finish_reason(r)
  if r == "STOP" then return "stop"
  elseif r == "MAX_TOKENS" then return "length"
  elseif r == "SAFETY" or r == "RECITATION" then return "content_filter"
  else return "stop" end
end

-- Split off system messages; merge consecutive same-role messages into
-- Google contents. Gemini wants alternating user/model turns.
local function build_contents(messages)
  local system_parts = {}
  local contents = {}
  local current = nil
  local current_role = nil
  local function flush()
    if current ~= nil then
      table.insert(contents, current)
      current = nil
    end
  end
  for _, m in ipairs(messages or {}) do
    if m.role == "system" then
      if m.content and m.content ~= "" then
        table.insert(system_parts, { text = m.content })
      end
    else
      local role = "user"
      if m.role == "assistant" then role = "model" end
      if current == nil or current_role ~= role then
        flush()
        current = { role = role, parts = {} }
        current_role = role
      end
      table.insert(current.parts, { text = m.content or "" })
    end
  end
  flush()
  return contents, system_parts
end

local function build_payload(request)
  local contents, system_parts = build_contents(request.messages)
  local payload = { contents = contents }
  if #system_parts > 0 then
    payload.systemInstruction = { parts = system_parts }
  end
  local cfg = nil
  if request.max_tokens and request.max_tokens > 0 then
    cfg = cfg or {}
    cfg.maxOutputTokens = request.max_tokens
  end
  if request.temperature and request.temperature > 0 then
    cfg = cfg or {}
    cfg.temperature = request.temperature
  end
  if request.top_p and request.top_p > 0 then
    cfg = cfg or {}
    cfg.topP = request.top_p
  end
  if request.reasoning_effort and request.reasoning_effort ~= "" and request.reasoning_effort ~= "minimal" then
    cfg = cfg or {}
    cfg.thinkingConfig = { includeThoughts = true }
  end
  local rf = request.response_format
  if type(rf) == "table" and type(rf.type) == "string" then
    if rf.type == "json_object" then
      cfg = cfg or {}
      cfg.responseMimeType = "application/json"
    elseif rf.type == "json_schema" then
      cfg = cfg or {}
      cfg.responseMimeType = "application/json"
      if type(rf.json_schema) == "table" and type(rf.json_schema.schema) == "table" then
        cfg.responseSchema = rf.json_schema.schema
      end
    end
  end
  if cfg then payload.generationConfig = cfg end
  return payload
end

-- Split candidate parts into visible text and thought summaries.
local function split_parts(cand)
  local text, reasoning = {}, {}
  local parts = cand and cand.content and cand.content.parts or {}
  for _, p in ipairs(parts) do
    if type(p.text) == "string" then
      if p.thought == true then
        table.insert(reasoning, p.text)
      else
        table.insert(text, p.text)
      end
    end
  end
  return table.concat(text), table.concat(reasoning)
end

local function build_usage(meta)
  if not meta then return nil end
  local usage = {
    prompt_tokens = meta.promptTokenCount or 0,
    completion_tokens = meta.candidatesTokenCount or 0,
    total_tokens = meta.totalTokenCount or 0,
  }
  if meta.thoughtsTokenCount and meta.thoughtsTokenCount > 0 then
    usage.completion_tokens_details = { reasoning_tokens = meta.thoughtsTokenCount }
  end
  if meta.cachedContentTokenCount and meta.cachedContentTokenCount > 0 then
    usage.prompt_tokens_details = { cached_tokens = meta.cachedContentTokenCount }
  end
  return usage
end

local function estimate_rpm(name)
  local m = name:lower()
  if m:find("flash") then
    if m:find("lite") then return 2000 end
    return 1000
  end
  if m:find("pro") then return 500 end
  return 1000
end

local function estimate_tpm(name)
  local m = name:lower()
  if m:find("flash") then
    if m:find("lite") then return 2000000 end
    return 1000000
  end
  if m:find("pro") then return 500000 end
  return 1000000
end

local function estimate_rpd(name)
  local m = name:lower()
  if m:find("flash") then
    if m:find("lite") then return 3000 end
    return 1500
  end
  if m:find("pro") then return 1000 end
  return 1500
end

local function model_short_name(entry)
  if type(entry.name) == "string" and entry.name ~= "" then
    return entry.name:gsub("^models/", "")
  end
  if type(entry.baseModelId) == "string" and entry.baseModelId ~= "" then
    return entry.baseModelId
  end
  return "unknown-model"
end

local function supports_generate_content(entry)
  for _, m in ipairs(entry.supportedGenerationMethods or {}) do
    if m == "generateContent" then return true end
  end
  return false
end

local function request_model_name(model)
  local name = model:match("([^/]+)$") or model
  return (name:gsub("^models/", ""))
end

llm_router.register("google", {
  icon = "https://www.gstatic.com/lamda/images/favicon_v1_150160cddff7f294ce30.svg",

  credential_schema = function()
    return {
      { type = "section", title = "Google AI Studio",
        subtitle = "Get your API key from Google AI Studio.",
        content = {
          { type = "link", text = "Open AI Studio", url = "https://aistudio.google.com/app/apikey" },
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
    local client = llm_router.create_http_client({})
    local key = api_key_of(credential)
    local url = BASE_URL .. "/models"
    local infos = {}
    while url do
      local resp, err = client:request({
        method = "GET", url = url,
        headers = { ["x-goog-api-key"] = key },
      })
      if err then return nil, err end
      if resp.status ~= 200 then return classify_error(resp.status, resp.body) end
      local page = json.decode(resp.body)
      for _, entry in ipairs(page.models or {}) do
        local name = model_short_name(entry)
        local skip = false
        if not supports_generate_content(entry) then skip = true end
        local lower = name:lower()
        if lower:find("embedding") or lower:find("aqa") then skip = true end
        if not skip then
          local display = entry.displayName
          if display == nil or display == "" then display = name end
          table.insert(infos, {
            name = name, display_name = display,
            rpm = estimate_rpm(name), tpm = estimate_tpm(name), rpd = estimate_rpd(name),
            context_window = entry.inputTokenLimit or 0,
            max_tokens = entry.outputTokenLimit or 0,
            supported_parameters = { "response_format", "structured_outputs" },
          })
          if entry.thinking == true then
            infos[#infos].reasoning = { default_enabled = true, supported_efforts = { "high", "medium", "low" } }
          end
        end
      end
      if page.nextPageToken and page.nextPageToken ~= "" then
        url = BASE_URL .. "/models?pageToken=" .. page.nextPageToken
      else
        url = nil
      end
    end
    return infos
  end,

  complete = function(ctx, credential, request)
    local client = llm_router.create_http_client({})
    local payload = build_payload(request)
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/models/" .. request_model_name(request.model) .. ":generateContent",
      headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(payload),
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.body) end
    local g = json.decode(resp.body)
    if (not g.candidates or #g.candidates == 0) and g.promptFeedback and g.promptFeedback.blockReason then
      return nil, { type = "invalid_request", message = "prompt blocked: " .. tostring(g.promptFeedback.blockReason) }
    end
    local choices = {}
    if g.candidates and #g.candidates > 0 then
      local cand = g.candidates[1]
      local text, reasoning = split_parts(cand)
      local message = { role = "assistant", content = text }
      if reasoning ~= "" then message.reasoning_content = reasoning end
      table.insert(choices, {
        index = cand.index or 0,
        message = message,
        finish_reason = map_finish_reason(cand.finishReason or ""),
      })
    end
    return {
      id = "google-" .. tostring(os.time()), object = "chat.completion", created = os.time(),
      model = request.model, choices = choices,
      usage = build_usage(g.usageMetadata) or { prompt_tokens = 0, completion_tokens = 0, total_tokens = 0 },
    }
  end,

  complete_stream = function(ctx, credential, request, emit)
    local client = llm_router.create_http_client({})
    local payload = build_payload(request)
    local full_model = request.model
    local chunk_id = "google-" .. tostring(os.time())
    local stream_err = client:stream({
      method = "POST",
      url = BASE_URL .. "/models/" .. request_model_name(request.model) .. ":streamGenerateContent?alt=sse",
      headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(payload),
      on_line = function(line)
        if line:sub(1, 6) ~= "data: " then return end
        local data = line:sub(7)
        if data == "[DONE]" then return end
        local ok, g = pcall(json.decode, data)
        if not ok or type(g) ~= "table" then return end
        local cand = g.candidates and g.candidates[1]
        if cand then
          local text, reasoning = split_parts(cand)
          local delta = {}
          if reasoning ~= "" then delta.reasoning_content = reasoning end
          if text ~= "" then delta.content = text end
          local choice = { index = cand.index or 0, delta = delta }
          if cand.finishReason and cand.finishReason ~= "" then
            choice.finish_reason = map_finish_reason(cand.finishReason)
          end
          if delta.content or delta.reasoning_content or choice.finish_reason then
            emit({
              id = chunk_id, object = "chat.completion.chunk", created = os.time(),
              model = full_model, choices = { choice },
            })
          end
        end
        local usage = build_usage(g.usageMetadata)
        if usage then
          emit({
            id = chunk_id, object = "chat.completion.chunk", created = os.time(),
            model = full_model, choices = {}, usage = usage,
          })
        end
      end,
    })
    if stream_err then
      return nil, stream_err
    end
  end,
})
