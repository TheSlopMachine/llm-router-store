--- @plugin OpenCode Zen
--- @author TheSlopMachine
--- @version 3.0.0
--- @router_version 0.1.1
--- @description OpenAI/Anthropic/Google compatible paid provider OpenCode Zen (API key required)
--- @allow_host opencode.ai

local BASE_URL = "https://opencode.ai/zen/v1"

-- Protocol families first: muse-spark/gpt/grok serve /responses even for
-- their -free variants. Remaining -free models are chat-protocol.
local function endpoint_for_model(model)
  local m = model:lower()
  if m:match("^gpt%-") or m:match("muse%-spark") or m:match("^grok%-") then
    return "/responses"
  end
  return "/chat/completions"
end

local MUSE_SPARK_MIN_OUTPUT_TOKENS = 512

local function responses_max_output_tokens(request, model)
  local value = tonumber(request.max_completion_tokens)
  if value == nil or value <= 0 then value = tonumber(request.max_tokens) end
  if value == nil or value <= 0 then return 32000 end
  if model:lower():match("^muse%-spark") and value < MUSE_SPARK_MIN_OUTPUT_TOKENS then
    return MUSE_SPARK_MIN_OUTPUT_TOKENS
  end
  return value
end

-- Keyed requests carry the genuine client wire shape: released-version
-- User-Agent plus ULID-shaped x-opencode-* session headers.
local GOOD_UA = "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
local ULID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
local HEXD = "0123456789abcdef"

local function hex12(n)
  local out = {}
  for i = 12, 1, -1 do
    local d = n % 16
    out[i] = HEXD:sub(d + 1, d + 1)
    n = (n - d) / 16
  end
  return table.concat(out)
end

local function time_head(descending, ms, counter)
  local cur = ms * 4096 + counter
  if descending then cur = 281474976710655 - cur end
  return hex12(cur)
end

local function rand_bytes(n)
  local ok, hex = pcall(llm_router.random_hex, n)
  if not ok or type(hex) ~= "string" or #hex < n * 2 then
    error("opencode-zen: llm_router.random_hex(" .. tostring(n) .. ") failed")
  end
  return hex
end

local function ulid_tail()
  local hex = rand_bytes(14)
  local out = {}
  for i = 1, 14 do
    local byte = tonumber(hex:sub(i * 2 - 1, i * 2), 16) or 0
    local c = byte % 62
    out[i] = ULID_CHARS:sub(c + 1, c + 1)
  end
  return table.concat(out)
end

-- Fresh ULID-shaped session/message ids per request. The time head uses
-- the current second plus a uniform sub-second part, matching the genuine
-- millisecond distribution; the message counter follows the session
-- counter, preserving request ordering. Shape match is sufficient: the
-- values carry no server-side state.
local function mint_ids()
  local jitter = tonumber(rand_bytes(2):sub(1, 4), 16) or 0
  local ms = os.time() * 1000 + (jitter % 1000)
  local ses = "ses_" .. time_head(true, ms, 1) .. ulid_tail()
  local msg = "msg_" .. time_head(false, ms, 2) .. ulid_tail()
  return ses, msg
end

local function opencode_headers(api_key, ses, msg)
  local auth = "Bearer " .. api_key
  return {
    ["User-Agent"] = GOOD_UA,
    ["Content-Type"] = "application/json",
    ["Accept"] = "application/json",
    ["Authorization"] = auth,
    ["x-opencode-client"] = "cli",
    ["x-opencode-project"] = "global",
    ["x-opencode-session"] = ses,
    ["x-opencode-request"] = msg,
  }
end

-- Plain-text view of a chat message. Content arrives either as a string
-- or as an array of content parts (multimodal/tool messages).
local function message_text(m)
  local content = m.content
  if type(content) == "string" then return content end
  if type(content) ~= "table" then return "" end
  local parts = {}
  for _, p in ipairs(content) do
    if type(p) == "string" then
      if p ~= "" then table.insert(parts, p) end
    elseif type(p) == "table" and type(p.text) == "string" and p.text ~= "" then
      table.insert(parts, p.text)
    end
  end
  return table.concat(parts, "\n")
end

-- Paid-tier limit semantics: quota wording binds to the account, plain rate
-- limits bind to the exit IP. Delay parsing lives in the core helper.
local function classify_extension(raw, default_err)
  if raw.status == 426 then
    local message = default_err and default_err.message or tostring(raw.body)
    return { type = "auth", message = message }
  end
  if raw.status ~= 429 then return nil end
  local err = default_err or { type = "rate_limit", message = tostring(raw.body) }
  local etype = ""
  do
    local ok, parsed = pcall(json.decode, tostring(raw.body))
    if ok and parsed and type(parsed.error) == "table" and type(parsed.error.type) == "string" then
      etype = parsed.error.type
    end
  end
  local lower_message = string.lower(err.message or "")
  local quota = string.lower(etype) == "freeusagelimiterror"
    or string.find(lower_message, "quota", 1, true) ~= nil
    or string.find(lower_message, "free usage", 1, true) ~= nil
  if quota then
    err.type = "quota_exceeded"
    err.scope = { "account" }
  else
    err.scope = { "proxy" }
  end
  return err
end

local function api_key_of(credential)
  if credential == nil or credential.data == nil then return "" end
  return credential.data.api_key or credential.data.access_token or ""
end

-- Paid-only provider: keyless credentials fail closed before any network.
local function require_api_key(credential)
  local key = api_key_of(credential)
  if key == "" then
    return nil, { type = "invalid_request", message = "api_key is required" }
  end
  return key
end

-- Paid models only: the keyed /models listing serves account entitlements.
-- This fallback mirrors it when the listing fails.
local FALLBACK_MODELS = {
  { name = "gpt-5", display_name = "GPT-5" },
  { name = "claude-sonnet-4.5", display_name = "Claude Sonnet 4.5" },
  { name = "gemini-3-flash", display_name = "Gemini 3 Flash" },
}

-- Upstream /models carries no capability metadata, so reasoning support
-- is matched by model family. Conservative: known-thinking families only.
-- (Declared before with_limits: Lua binds the name used inside a function
-- at compile time, so a later local would resolve to a nil global.)
local function supports_reasoning(name)
  local m = name:lower()
  if m:find("claude") then return true end
  if m:find("^gpt%-5") or m:match("^o[0-9]") then return true end
  if m:find("gemini") and not m:find("tts") and not m:find("image") then return true end
  if m:find("grok%-4") or m:find("grok%-code") then return true end
  return false
end

local function with_limits(infos)
  for _, m in ipairs(infos) do
    m.context_window = 200000
    m.max_tokens = 32000
    m.rpm = 60
    m.tpm = 100000
    m.rpd = 500
    m.supported_parameters = { "tools", "tool_choice", "response_format", "temperature", "top_p", "max_tokens" }
    if supports_reasoning(m.name or "") then
      m.reasoning = { default_enabled = true, supported_efforts = { "high", "medium", "low" } }
    end
  end
  return infos
end

-- Build the OpenAI Responses input from chat messages.
local function build_responses_input(messages)
  local input = {}
  for _, m in ipairs(messages or {}) do
    local role = m.role
    if role == "system" then
      local text = message_text(m)
      if text ~= "" then
        table.insert(input, { role = "system", content = text })
      end
    elseif role == "user" then
      local text = message_text(m)
      if text ~= "" then
        table.insert(input, { role = "user", content = { { type = "input_text", text = text } } })
      end
    elseif role == "assistant" then
      local text = message_text(m)
      if text ~= "" then
        table.insert(input, { role = "assistant", content = { { type = "output_text", text = text } } })
      end
      for _, tc in ipairs(m.tool_calls or {}) do
        local name = tc["function"] and tc["function"].name or ""
        if name ~= "" then
          local args = tc["function"].arguments or "{}"
          if args == "" then args = "{}" end
          table.insert(input, { type = "function_call", call_id = tc.id or "", name = name, arguments = args })
        end
      end
    elseif role == "tool" then
      table.insert(input, { type = "function_call_output", call_id = m.tool_call_id or "call_unknown", output = message_text(m) })
    end
  end
  if #input == 0 then
    table.insert(input, { role = "user", content = { { type = "input_text", text = "ping" } } })
  end
  return input
end

local function build_responses_tools(tools)
  local out = {}
  for _, t in ipairs(tools or {}) do
    local name = t.name or ""
    local desc = t.description or ""
    local params = t.parameters
    if t["function"] then
      if t["function"].name and t["function"].name ~= "" then name = t["function"].name end
      if t["function"].description and t["function"].description ~= "" then desc = t["function"].description end
      if t["function"].parameters then params = t["function"].parameters end
    end
    if name ~= "" then
      if params == nil then params = { type = "object", properties = {} } end
      table.insert(out, { type = "function", name = name, description = desc, parameters = params })
    end
  end
  return out
end

local function extract_responses_text(raw)
  local output = raw.output
  if type(output) == "table" then
    for _, item in ipairs(output) do
      if type(item) == "table" and item.type == "message" and type(item.content) == "table" then
        for _, c in ipairs(item.content) do
          if type(c) == "table" then
            if type(c.text) == "string" and c.text ~= "" then return c.text end
            if type(c.output_text) == "string" and c.output_text ~= "" then return c.output_text end
          end
        end
      end
    end
  end
  if type(raw.output_text) == "string" then return raw.output_text end
  return ""
end

local function extract_responses_reasoning(raw)
  local output = raw.output
  if type(output) ~= "table" then return "" end
  local parts = {}
  for _, item in ipairs(output) do
    if type(item) == "table" and item.type == "reasoning" and type(item.summary) == "table" then
      for _, s in ipairs(item.summary) do
        if type(s) == "table" and type(s.text) == "string" and s.text ~= "" then
          table.insert(parts, s.text)
        end
      end
    end
  end
  return table.concat(parts, "\n")
end

local function extract_responses_tool_calls(raw)
  local out = {}
  local output = raw.output
  if type(output) ~= "table" then return out end
  for _, item in ipairs(output) do
    if type(item) == "table" and item.type == "function_call" and type(item.name) == "string" and item.name ~= "" then
      local args = item.arguments or "{}"
      if type(args) == "table" then args = json.encode(args) end
      if args == "" then args = "{}" end
      local id = item.call_id or item.id or ""
      table.insert(out, { id = id, type = "function", ["function"] = { name = item.name, arguments = args } })
    end
  end
  return out
end

llm_router.register("opencode-zen", {
  icon = "https://opencode.ai/favicon.ico",

  classify_error = classify_extension,

  credential_schema = function()
    return {
      { type = "section", title = "OpenCode Zen",
        content = {
          { type = "banner", variant = "info",
            text = "Zen API key required. Paid models only." },
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
    local key, key_err = require_api_key(credential)
    if key_err then return nil, key_err end
    local client = llm_router.http_client({})
    local ses, msg = mint_ids()
    local headers = opencode_headers(key, ses, msg)
    local resp, err = client:request({
      method = "GET", url = BASE_URL .. "/models",
      headers = headers,
    })
    if err then
      -- Upstream listing failed: fall back to the known model set.
      return with_limits(FALLBACK_MODELS)
    end
    if resp.status ~= 200 then
      return with_limits(FALLBACK_MODELS)
    end
    local ok, parsed = pcall(json.decode, resp.body)
    if not ok or not parsed or type(parsed.data) ~= "table" then
      return with_limits(FALLBACK_MODELS)
    end
    local infos = {}
    for _, m in ipairs(parsed.data) do
      if type(m.id) == "string" and m.id ~= "" then
        table.insert(infos, { name = m.id, display_name = m.id })
      end
    end
    if #infos == 0 then
      return with_limits(FALLBACK_MODELS)
    end
    return with_limits(infos)
  end,

  complete = function(ctx, credential, request)
    local model = request.model_name
    local endpoint = endpoint_for_model(model)
    local client = llm_router.http_client({})

    local api_key, key_err = require_api_key(credential)
    if key_err then return nil, key_err end
    local ses, msg = mint_ids()

    if endpoint == "/responses" then
      local payload = { model = model, input = build_responses_input(request.messages), stream = false }
      payload.max_output_tokens = responses_max_output_tokens(request, model)
      if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
      if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
      local effort = request.reasoning_effort
      if effort == "low" or effort == "medium" or effort == "high" or effort == "xhigh" then
        payload.reasoning = { effort = effort, summary = "auto" }
      end
      local rf = request.response_format
      if type(rf) == "table" and type(rf.type) == "string" then
        if rf.type == "json_object" then
          payload.text = { format = { type = "json_object" } }
        elseif rf.type == "json_schema" and type(rf.json_schema) == "table" then
          local fmt = { type = "json_schema", strict = true }
          if type(rf.json_schema.name) == "string" then fmt.name = rf.json_schema.name end
          if type(rf.json_schema.schema) == "table" then fmt.schema = rf.json_schema.schema end
          payload.text = { format = fmt }
        end
      end
      local tools = build_responses_tools(request.tools)
      if #tools > 0 then payload.tools = tools end

      local headers = opencode_headers(api_key, ses, msg)
      local resp, err = client:request({
        method = "POST", url = BASE_URL .. "/responses",
        headers = headers, body = json.encode(payload),
      })
      if err then return nil, err end
      if resp.status ~= 200 then
        return nil, llm_router.classify_error({ status = resp.status, headers = resp.headers, body = resp.body })
      end
      local raw = json.decode(resp.body)
      local text = extract_responses_text(raw)
      local reasoning = extract_responses_reasoning(raw)
      local tool_calls = extract_responses_tool_calls(raw)
      local finish = "stop"
      if type(raw.incomplete_details) == "table"
          and raw.incomplete_details.reason == "max_output_tokens" then
        finish = "length"
      end
      if #tool_calls > 0 then finish = "tool_calls" end
      local usage = { prompt_tokens = 0, completion_tokens = 0, total_tokens = 0 }
      if type(raw.usage) == "table" then
        usage.prompt_tokens = raw.usage.input_tokens or 0
        usage.completion_tokens = raw.usage.output_tokens or 0
        usage.total_tokens = raw.usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
      end
      local message = { role = "assistant", content = text, tool_calls = tool_calls }
      if reasoning ~= "" then message.reasoning_content = reasoning end
      return {
        id = "zen-" .. tostring(os.time()), object = "chat.completion", created = os.time(),
        model = request.model,
        choices = {
          { index = 0, message = message, finish_reason = finish },
        },
        usage = usage,
      }
    end

    local payload = { model = model, messages = request.messages, stream = false }
    if request.max_tokens and request.max_tokens > 0 then payload.max_tokens = request.max_tokens end
    if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
    if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
    if request.tools then payload.tools = request.tools end
    if request.tool_choice then payload.tool_choice = request.tool_choice end
    if type(request.response_format) == "table" then payload.response_format = request.response_format end

    local headers = opencode_headers(api_key, ses, msg)
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. endpoint,
      headers = headers,
      body = json.encode(payload),
      on_response = function(r)
        if r.status ~= 200 then
          return llm_router.classify_error({ status = r.status, headers = r.headers, body = r.body })
        end
      end,
    })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, llm_router.classify_error({ status = resp.status, headers = resp.headers, body = resp.body })
    end
    local out = json.decode(resp.body)
    if out.model == nil or out.model == "" then out.model = request.model end
    return out
  end,

  complete_stream = function(ctx, credential, request, emit)
    local model = request.model_name
    local api_key, key_err = require_api_key(credential)
    if key_err then return nil, key_err end
    local ses, msg = mint_ids()
    local client = llm_router.http_client({})
    local full_model = request.model

    -- Keyed chat path: relay the client's shape.
    local endpoint = endpoint_for_model(model)
    if endpoint == "/responses" then
      local payload = { model = model, input = build_responses_input(request.messages), stream = true }
      payload.max_output_tokens = responses_max_output_tokens(request, model)
      if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
      if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
      local effort = request.reasoning_effort
      if effort == "low" or effort == "medium" or effort == "high" or effort == "xhigh"
          or effort == "minimal" or effort == "max" then
        payload.reasoning = { effort = effort, summary = "auto" }
      end
      local tools = build_responses_tools(request.tools)
      if #tools > 0 then
        payload.tools = tools
        if request.tool_choice then payload.tool_choice = request.tool_choice end
      end
      local headers = opencode_headers(api_key, ses, msg)
      local _, stream_err = client:stream({
        method = "POST", url = BASE_URL .. "/responses",
        headers = headers,
        body = json.encode(payload),
        on_response = function(r)
          if r.status ~= 200 then
            return llm_router.classify_error({ status = r.status, headers = r.headers, body = r.body })
          end
        end,
        on_line = function(line)
          if line:sub(1, 6) ~= "data: " then return end
          local data = line:sub(7)
          if data == "[DONE]" or data == "" then return end
          local ok, chunk = pcall(json.decode, data)
          if not ok or type(chunk) ~= "table" then return end
          emit(chunk)
        end,
      })
      if stream_err then
        return nil, stream_err
      end
      return
    end

    local payload = { model = model, messages = request.messages, stream = true }
    if request.max_tokens and request.max_tokens > 0 then payload.max_tokens = request.max_tokens end
    if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
    if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
    if request.tools then payload.tools = request.tools end
    if request.tool_choice then payload.tool_choice = request.tool_choice end
    if type(request.response_format) == "table" then payload.response_format = request.response_format end
    local headers = opencode_headers(api_key, ses, msg)
    local _, stream_err = client:stream({
      method = "POST", url = BASE_URL .. endpoint,
      headers = headers,
      body = json.encode(payload),
      on_line = function(line)
        if line:sub(1, 6) ~= "data: " then return end
        local data = line:sub(7)
        if data == "[DONE]" or data == "" then return end
        local ok, chunk = pcall(json.decode, data)
        if not ok or type(chunk) ~= "table" then return end
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

})
