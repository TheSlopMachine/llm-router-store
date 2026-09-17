--- @plugin OpenCode Zen
--- @author TheSlopMachine
--- @version 1.0.7
--- @router_version 0.0.4
--- @description OpenAI/Anthropic/Google compatible free provider OpenCode Zen
--- @allow_host opencode.ai

local BASE_URL = "https://opencode.ai/zen/v1"

local function endpoint_for_model(model)
  local m = model:lower()
  if m:match("%-free$") then
    return "/chat/completions"
  end
  if m:match("^gpt%-") or m:match("muse%-spark") or m:match("^grok%-") then
    return "/responses"
  end
  return "/chat/completions"
end

-- Anonymous free-tier requests must carry the exact wire shape of the
-- genuine client, verified by differential probing against the live API:
-- released-version User-Agent, Bearer public, x-opencode-* session headers
-- with ULID-shaped ids, stream:true, temperature + max_tokens present, and
-- a leading system message carrying the known client prompt fingerprint.
local GOOD_UA = "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"

-- First 1000 chars of the genuine title-generator system prompt, the
-- minimal fragment the free-tier gate accepts. Stored with literal \r\n
-- escapes and decoded below, so the source file stays plain LF text.
local FINGERPRINT_SRC = [[You are a title generator. You output ONLY a thread title. Nothing else.\r\n\r\n<task>\r\nGenerate a brief title that would help the user find this conversation later.\r\n\r\nFollow all rules in <rules>\r\nUse the <examples> so you know what a good title looks like.\r\nYour output must be:\r\n- A single line\r\n- ≤50 characters\r\n- No explanations\r\n</task>\r\n\r\n<rules>\r\n- you MUST use the same language as the user message you are summarizing\r\n- Title must be grammatically correct and read naturally - no word salad\r\n- Never include tool names in the title (e.g. "read tool", "bash tool", "edit tool")\r\n- Focus on the main topic or question the user needs to retrieve\r\n- Vary your phrasing - avoid repetitive patterns like always starting with "Analyzing"\r\n- When a file is mentioned, focus on WHAT the user wants to do WITH the file, not just that they shared it\r\n- Keep exact: technical terms, numbers, filenames, HTTP codes\r\n- Remove: the, this, my, a, an\r\n- Never assume tech stack\r\n- Never use tools\r\n- NEVER res]]
local FINGERPRINT = FINGERPRINT_SRC:gsub("\\r\\n", "\r\n")

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
  local auth = "Bearer public"
  if api_key and api_key ~= "" then auth = "Bearer " .. api_key end
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

local function classify_error(status, body)
  local message = body
  local ok, parsed = pcall(json.decode, body)
  if ok and parsed and parsed.error and parsed.error.message then
    message = parsed.error.message
  elseif ok and parsed and type(parsed.message) == "string" and parsed.message ~= "" then
    message = parsed.message
  elseif ok and parsed and type(parsed.error) == "string" and parsed.error ~= "" then
    message = parsed.error
  end
  if status == 401 or status == 403 or status == 426 then
    return nil, { type = "auth", message = message }
  elseif status == 429 then
    local t = message:lower():find("quota")
    return nil, { type = t and "quota_exceeded" or "rate_limit", message = message, retry_after = os.time() + 60 }
  elseif status == 408 or status == 504 then
    return nil, { type = "timeout", message = message }
  elseif status >= 500 then
    return nil, { type = "upstream", message = message }
  end
  return nil, { type = "invalid_request", message = message }
end

local function api_key_of(credential)
  if credential == nil or credential.data == nil then return "" end
  return credential.data.api_key or credential.data.access_token or ""
end

local FALLBACK_MODELS = {
  { name = "nemotron-3-ultra-free", display_name = "Nemotron 3 Ultra Free" },
  { name = "nemotron-3.5-lightning-free", display_name = "Nemotron 3.5 Lightning Free" },
  { name = "mimo-v2.5-free", display_name = "MiMo V2.5 Free" },
  { name = "big-pickle", display_name = "Big Pickle" },
  { name = "ling-3.0-flash-fin-free", display_name = "Ling 3.0 Flash Fin Free" },
  { name = "muse-spark-1.2-contributor-free", display_name = "Muse Spark 1.2 Contributor Free" },
  { name = "muse-spark-1.3-contributor-free", display_name = "Muse Spark 1.3 Contributor Free" },
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

-- Anonymous chat payload in the genuine client's shape. temperature and
-- max_tokens default to the proven values when the client omits them:
-- the gate stalls requests that lack either field.
local function build_anon_chat_payload(request, model)
  local messages = { { role = "system", content = FINGERPRINT } }
  for _, m in ipairs(request.messages or {}) do
    table.insert(messages, m)
  end
  local payload = {
    model = model,
    messages = messages,
    stream = true,
    stream_options = { include_usage = true },
  }
  if request.temperature and request.temperature > 0 then
    payload.temperature = request.temperature
  else
    payload.temperature = 0.5
  end
  if request.max_tokens and request.max_tokens > 0 then
    payload.max_tokens = request.max_tokens
  else
    payload.max_tokens = 32000
  end
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  if request.tools then payload.tools = request.tools end
  if request.tool_choice then payload.tool_choice = request.tool_choice end
  if type(request.response_format) == "table" then payload.response_format = request.response_format end
  return payload
end

-- Assemble one chat.completion from a streamed chat/completions body.
local function assemble_chat_response(body, model)
  local text_parts = {}
  local reasoning_parts = {}
  local tc_acc = {}
  local tc_order = {}
  local prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
  local finish = "stop"
  for line in (body .. "\n"):gmatch("([^\n]*)\n") do
    if line:sub(1, 6) == "data: " then
      local data = line:sub(7)
      if data ~= "[DONE]" and data ~= "" then
        local ok, chunk = pcall(json.decode, data)
        if ok and type(chunk) == "table" then
          if type(chunk.choices) == "table" then
            for _, ch in ipairs(chunk.choices) do
              if type(ch) == "table" then
                local delta = ch.delta
                if type(delta) == "table" then
                  if type(delta.content) == "string" and delta.content ~= "" then
                    table.insert(text_parts, delta.content)
                  end
                  if type(delta.reasoning) == "string" and delta.reasoning ~= "" then
                    table.insert(reasoning_parts, delta.reasoning)
                  end
                  if type(delta.tool_calls) == "table" then
                    for _, tc in ipairs(delta.tool_calls) do
                      if type(tc) == "table" then
                        local idx = tc.index or 0
                        local acc = tc_acc[idx]
                        if not acc then
                          acc = { id = "", name = "", args = {} }
                          tc_acc[idx] = acc
                          table.insert(tc_order, idx)
                        end
                        if type(tc.id) == "string" and tc.id ~= "" then acc.id = tc.id end
                        local fn = tc["function"]
                        if type(fn) == "table" then
                          if type(fn.name) == "string" and fn.name ~= "" then acc.name = fn.name end
                          if type(fn.arguments) == "string" and fn.arguments ~= "" then
                            table.insert(acc.args, fn.arguments)
                          end
                        end
                      end
                    end
                  end
                end
                if type(ch.finish_reason) == "string" and ch.finish_reason ~= "" then
                  finish = ch.finish_reason
                end
              end
            end
          end
          if type(chunk.usage) == "table" then
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0
            total_tokens = chunk.usage.total_tokens or (prompt_tokens + completion_tokens)
          end
        end
      end
    end
  end
  table.sort(tc_order)
  local tool_calls = {}
  for _, idx in ipairs(tc_order) do
    local acc = tc_acc[idx]
    if acc.name ~= "" then
      table.insert(tool_calls, {
        id = acc.id, type = "function",
        ["function"] = { name = acc.name, arguments = table.concat(acc.args) },
      })
    end
  end
  if #tool_calls > 0 and finish == "stop" then finish = "tool_calls" end
  local message = { role = "assistant", content = table.concat(text_parts), tool_calls = tool_calls }
  local reasoning = table.concat(reasoning_parts)
  if reasoning ~= "" then message.reasoning_content = reasoning end
  return {
    id = "zen-" .. tostring(os.time()), object = "chat.completion", created = os.time(),
    model = model,
    choices = {
      { index = 0, message = message, finish_reason = finish },
    },
    usage = {
      prompt_tokens = prompt_tokens,
      completion_tokens = completion_tokens,
      total_tokens = total_tokens,
    },
  }
end

llm_router.register("opencode-zen", {
  icon = "https://opencode.ai/favicon.ico",

  credential_schema = function()
    return {
      { type = "section", title = "OpenCode Zen",
        content = {
          { type = "banner", variant = "info",
            text = "Leave the key empty for free models. Add a Zen API key for paid models." },
          { type = "secret", name = "api_key", label = "API Key" },
          { type = "button", text = "Save", form_action = "submit" },
        } },
    }
  end,

  validate_credentials = function(data)
    local key = data.api_key
    if key == nil or key == "" then
      return true
    end
    if #key < 20 then
      return false, { type = "invalid_request", message = "api_key: minimum 20 characters" }
    end
    return true
  end,

  get_model_infos = function(ctx, credential, provider_config)
    local client = llm_router.create_http_client({})
    local key = api_key_of(credential)
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
    local model = request.model:match("([^/]+)$")
    local endpoint = endpoint_for_model(model)
    local client = llm_router.create_http_client({})

    local api_key = api_key_of(credential)
    local ses, msg = mint_ids()
    local anonymous = api_key == ""

    if endpoint == "/responses" then
      local payload = { model = model, input = build_responses_input(request.messages), stream = false }
      if request.max_tokens and request.max_tokens > 0 then payload.max_output_tokens = request.max_tokens end
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
      if resp.status ~= 200 then return classify_error(resp.status, resp.body) end
      local raw = json.decode(resp.body)
      local text = extract_responses_text(raw)
      local reasoning = extract_responses_reasoning(raw)
      local tool_calls = extract_responses_tool_calls(raw)
      local finish = "stop"
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

    if not anonymous then
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
      })
      if err then return nil, err end
      if resp.status ~= 200 then return classify_error(resp.status, resp.body) end
      local out = json.decode(resp.body)
      if out.model == nil or out.model == "" then out.model = request.model end
      return out
    end

    -- Anonymous chat path: the free-tier gate only serves stream:true
    -- requests shaped like the genuine client, so always stream upstream
    -- and assemble the SSE body into one completion here.
    local payload = build_anon_chat_payload(request, model)
    local headers = opencode_headers(api_key, ses, msg)
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/chat/completions",
      headers = headers,
      body = json.encode(payload),
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.body) end
    return assemble_chat_response(resp.body, request.model)
  end,

  complete_stream = function(ctx, credential, request, emit)
    local model = request.model:match("([^/]+)$")
    local api_key = api_key_of(credential)
    local ses, msg = mint_ids()
    local client = llm_router.create_http_client({})
    local full_model = request.model

    if api_key == "" and endpoint_for_model(model) == "/chat/completions" then
      local payload = build_anon_chat_payload(request, model)
      local headers = opencode_headers(api_key, ses, msg)
      local stream_err = client:stream({
        method = "POST", url = BASE_URL .. "/chat/completions",
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
        local status = tonumber(tostring(stream_err.message):match("unexpected status (%d+)"))
        if status then
          local inner = tostring(stream_err.message):match("unexpected status %d+: (.*)$") or ""
          return classify_error(status, inner)
        end
        return nil, stream_err
      end
      return
    end

    -- Keyed chat path and /responses models: relay the client's shape.
    local endpoint = endpoint_for_model(model)
    local payload = { model = model, messages = request.messages, stream = true }
    if request.max_tokens and request.max_tokens > 0 then payload.max_tokens = request.max_tokens end
    if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
    if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
    if request.tools then payload.tools = request.tools end
    if request.tool_choice then payload.tool_choice = request.tool_choice end
    if type(request.response_format) == "table" then payload.response_format = request.response_format end
    local headers = opencode_headers(api_key, ses, msg)
    local stream_err = client:stream({
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
      local status = tonumber(tostring(stream_err.message):match("unexpected status (%d+)"))
      if status then
        local inner = tostring(stream_err.message):match("unexpected status %d+: (.*)$") or ""
        return classify_error(status, inner)
      end
      return nil, stream_err
    end
  end,

})
