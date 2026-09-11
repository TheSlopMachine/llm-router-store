--- @plugin Groq
--- @author TheSlopMachine
--- @version 1.1.0
--- @router_version 0.0.4
--- @description Groq OpenAI-compatible API: fast Llama/Qwen/gpt-oss inference
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
}
local DEFAULT_LIMITS = { rpm = 30, rpd = 1000, tpm = 8000 }

local function api_key_of(credential)
  if credential == nil or credential.data == nil then return "" end
  return credential.data.api_key or ""
end

-- Strip the router provider prefix: "groq/openai/gpt-oss-20b" -> "openai/gpt-oss-20b".
local function bare_model(model_id)
  return (tostring(model_id):gsub("^[^/]+/", ""))
end

local function classify_error(status, headers, body)
  local message = body
  local ok, parsed = pcall(json.decode, body)
  if ok and parsed and parsed.error and type(parsed.error.message) == "string" and parsed.error.message ~= "" then
    message = parsed.error.message
  end
  if status == 401 or status == 403 then
    return nil, { type = "auth", message = message }
  elseif status == 429 then
    local wait = 60
    if headers and type(headers["retry-after"]) == "string" then
      local n = tonumber(headers["retry-after"])
      if n and n > 0 then wait = math.ceil(n) end
    end
    -- Daily-quota exhaustion (RPD) rotates the credential; minute limits just back off.
    if message:find("per day") then
      return nil, { type = "quota_exceeded", message = message, retry_after = os.time() + wait }
    end
    return nil, { type = "rate_limit", message = message, retry_after = os.time() + wait }
  elseif status == 408 or status == 504 then
    return nil, { type = "timeout", message = message }
  elseif status >= 500 then
    return nil, { type = "upstream", message = message }
  end
  return nil, { type = "invalid_request", message = message }
end

-- Build the upstream payload from the OpenAI-shaped request table:
-- pass everything through, fix the model id, pin the stream flag.
local function build_payload(request, stream)
  local payload = {}
  for k, v in pairs(request) do
    payload[k] = v
  end
  payload.model = bare_model(request.model)
  payload.stream = stream
  if not stream then
    payload.stream_options = nil
  end
  return payload
end

llm_router.register("groq", {
  icon = "https://console.groq.com/favicon.ico",

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
    local client = llm_router.create_http_client({})
    local resp, err = client:request({
      method = "GET", url = BASE_URL .. "/models",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential) },
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
    local page = json.decode(resp.body)
    local infos = {}
    for _, m in ipairs(page.data or {}) do
      -- Chat models only: text in, text out. Drops whisper (audio) and orpheus (speech out).
      local function has(list, v)
        for _, x in ipairs(list or {}) do if x == v then return true end end
        return false
      end
      if m.active and has(m.input_modalities, "text") and has(m.output_modalities, "text") then
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
        })
      end
    end
    table.sort(infos, function(a, b) return a.name < b.name end)
    return infos
  end,

  complete = function(ctx, credential, request)
    local client = llm_router.create_http_client({})
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/chat/completions",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(build_payload(request, false)),
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
    local out = json.decode(resp.body)
    out.model = request.model
    return out
  end,

  complete_stream = function(ctx, credential, request, emit)
    local client = llm_router.create_http_client({})
    local full_model = request.model
    local stream_err = client:stream({
      method = "POST", url = BASE_URL .. "/chat/completions",
      headers = { ["Authorization"] = "Bearer " .. api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(build_payload(request, true)),
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
      -- client:stream wraps non-2xx as "unexpected status N: body"; re-classify.
      local status = tonumber(tostring(stream_err.message):match("unexpected status (%d+)"))
      if status then
        local body = tostring(stream_err.message)
        local inner = body:match("unexpected status %d+: (.*)$") or ""
        return classify_error(status, nil, inner)
      end
      return nil, stream_err
    end
  end,
})
