--- @plugin OpenCode Zen
--- @author TheSlopMachine
--- @version 6.0.0
--- @plugin_api 1.0
--- @description OpenAI/Anthropic/Google compatible paid provider OpenCode Zen (API key required)
--- @allow_host opencode.ai

local BASE_URL = "https://opencode.ai/zen/v1"

-- Protocol families first: muse-spark/gpt/grok serve /responses even for
-- their -free variants. Remaining -free models are chat-protocol.
local function endpoint_for_model(model)
  local m = model:lower()
  if m:match("^gpt%-") or m:match("muse%-spark") or m:match("^grok%-") then return "/responses" end
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

local function api_key_of(data)
  if type(data) ~= "table" then return "" end
  return data.api_key or data.access_token or ""
end

-- Dead-key bench: shared disable when automation is on (visible in the
-- dashboard, global to every path), unified cooldown park otherwise.
local function bench(ctx, cred_id, reason, wait_secs)
  local cfg = ctx.provider_config or {}
  if cfg.disable_failed_credentials == true then
    llm_router.credentials.disable(cred_id, reason)
  else
    llm_router.credentials.park(cred_id, wait_secs, reason)
  end
end

local function retry_after_secs(resp)
  local headers = resp.headers or {}
  local n = tonumber(headers["retry-after"] or headers["retry_after"])
  if n ~= nil and n > 0 then return math.floor(n) end
  return 60
end

local function pick_proxies(ctx, limit)
  -- Unconfigured providers go direct: only an explicit pool selection
  -- (dashboard proxy switch) routes through pooled exits.
  local pool = nil
  if ctx.provider_config and ctx.provider_config.proxy and ctx.provider_config.proxy.pool ~= "" then
    pool = ctx.provider_config.proxy.pool
  end
  -- The pool acquires exits on demand: it searches for a live proxy when none
  -- is ready and waits for one. A direct pool, the wait deadline or a
  -- cancelled request degrade to one direct attempt, never to silence.
  local res = llm_router.proxies.require({ pool = pool, limit = limit or 3, fallback = "direct" })
  if not res or #res.proxies == 0 then
    return { {} }
  end
  return res.proxies
end

-- Paid-tier limit semantics: quota wording binds to the account, plain rate
-- limits bind to the exit IP. Outcomes map to loop decisions: "proxy"
-- retries the same credential on the next exit, "cred" moves to the next
-- credential, "done" returns the terminal error to the client at once.
local function map_upstream(ctx, resp, cred_id)
  if resp.status == 426 then
    return "done", { message = "zen client upgrade required", code = "authentication_error", status = 426 }
  end
  if resp.status == 401 then
    bench(ctx, cred_id, "zen rejected the api key", 300)
    return "cred", { message = "zen rejected the api key", code = "authentication_error", status = 401 }
  end
  if resp.status ~= 429 then
    return "done",
      { message = "zen returned status " .. tostring(resp.status), code = "server_error", status = resp.status }
  end
  local body_str = tostring(resp.body or "")
  local etype = ""
  do
    local ok, parsed = pcall(json.decode, body_str)
    if ok and parsed and type(parsed.error) == "table" and type(parsed.error.type) == "string" then
      etype = parsed.error.type
    end
  end
  local lower_message = string.lower(body_str)
  local quota = string.lower(etype) == "freeusagelimiterror"
    or string.find(lower_message, "quota", 1, true) ~= nil
    or string.find(lower_message, "free usage", 1, true) ~= nil
  if quota then
    llm_router.credentials.park(cred_id, retry_after_secs(resp), "zen quota exhausted")
    return "cred", { message = "zen quota exhausted", code = "insufficient_quota", status = 429 }
  end
  return "proxy", { message = "zen rate limited on current exit", code = "rate_limit", status = 429 }
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
  if m == "gemini-3-flash" then return false end
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
    m.input_modalities = { "text" }
    m.output_modalities = { "text" }
    m.endpoints = { "chat/completions" }
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
      if text ~= "" then table.insert(input, { role = "system", content = text }) end
    elseif role == "user" then
      local text = message_text(m)
      if text ~= "" then table.insert(input, { role = "user", content = { { type = "input_text", text = text } } }) end
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
      table.insert(
        input,
        { type = "function_call_output", call_id = m.tool_call_id or "call_unknown", output = message_text(m) }
      )
    end
  end
  if #input == 0 then table.insert(input, { role = "user", content = { { type = "input_text", text = "ping" } } }) end
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
        if type(s) == "table" and type(s.text) == "string" and s.text ~= "" then table.insert(parts, s.text) end
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

-- One /responses attempt: returns (result) or (nil, action, err) where
-- action is "retry" (next proxy), "cred" (next credential) or "done".
local function do_responses_unary(client, request, model, api_key, ses, msg, proxy_url, cred_id)
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
    method = "POST",
    url = BASE_URL .. "/responses",
    headers = headers,
    body = json.encode(payload),
    proxy_url = proxy_url,
  })
  if err then return nil, "retry", err end
  if resp.status ~= 200 then
    local action, terr = map_upstream(ctx, resp, cred_id)
    return nil, action, terr
  end
  local raw = json.decode(resp.body)
  local text = extract_responses_text(raw)
  local reasoning = extract_responses_reasoning(raw)
  local tool_calls = extract_responses_tool_calls(raw)
  local finish = "stop"
  if type(raw.incomplete_details) == "table" and raw.incomplete_details.reason == "max_output_tokens" then
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
    id = "zen-" .. tostring(os.time()),
    object = "chat.completion",
    created = os.time(),
    model = request.model,
    choices = {
      { index = 0, message = message, finish_reason = finish },
    },
    usage = usage,
  }
end

local function do_chat_unary(client, request, model, endpoint, api_key, ses, msg, proxy_url, cred_id)
  local payload = { model = model, messages = request.messages, stream = false }
  if request.max_tokens and request.max_tokens > 0 then payload.max_tokens = request.max_tokens end
  if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  if request.tools then
    payload.tools = request.tools
  else
    payload.tools = {}
  end
  if request.tool_choice then payload.tool_choice = request.tool_choice end
  if type(request.response_format) == "table" then payload.response_format = request.response_format end
  local headers = opencode_headers(api_key, ses, msg)
  local resp, err = client:request({
    method = "POST",
    url = BASE_URL .. endpoint,
    headers = headers,
    body = json.encode(payload),
    proxy_url = proxy_url,
  })
  if err then return nil, "retry", err end
  if resp.status ~= 200 then
    local action, terr = map_upstream(ctx, resp, cred_id)
    return nil, action, terr
  end
  local out = json.decode(resp.body)
  if out.model == nil or out.model == "" then out.model = request.model end
  return out
end

-- One /responses stream attempt: relays upstream chunks directly and
-- returns nil on success or the terminal/transport error. st tracks
-- done (emitted), fatal (terminal, stop everything) and next_cred.
local function do_responses_stream(client, request, model, api_key, ses, msg, proxy_url, cred_id, emit, st)
  local payload = { model = model, input = build_responses_input(request.messages), stream = true }
  payload.max_output_tokens = responses_max_output_tokens(request, model)
  if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  local effort = request.reasoning_effort
  if
    effort == "low"
    or effort == "medium"
    or effort == "high"
    or effort == "xhigh"
    or effort == "minimal"
    or effort == "max"
  then
    payload.reasoning = { effort = effort, summary = "auto" }
  end
  local tools = build_responses_tools(request.tools)
  if #tools > 0 then
    payload.tools = tools
    if request.tool_choice then payload.tool_choice = request.tool_choice end
  end
  local headers = opencode_headers(api_key, ses, msg)
  local _, stream_err = client:stream({
    method = "POST",
    url = BASE_URL .. "/responses",
    headers = headers,
    body = json.encode(payload),
    proxy_url = proxy_url,
    on_response = function(r)
      if r.status == 200 then return end
      local action, terr = map_upstream(ctx, r, cred_id)
      if action == "done" then st.fatal = true end
      if action == "cred" then st.next_cred = true end
      return terr
    end,
    on_line = function(line)
      if line:sub(1, 6) ~= "data: " then return end
      local data = line:sub(7)
      if data == "[DONE]" or data == "" then return end
      local ok, chunk = pcall(json.decode, data)
      if not ok or type(chunk) ~= "table" then return end
      emit(chunk)
      st.done = true
    end,
  })
  return stream_err
end

local function do_chat_stream(
  client,
  request,
  model,
  endpoint,
  api_key,
  ses,
  msg,
  proxy_url,
  cred_id,
  full_model,
  emit,
  st
)
  local payload = { model = model, messages = request.messages, stream = true }
  if request.max_tokens and request.max_tokens > 0 then payload.max_tokens = request.max_tokens end
  if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  if request.tools then
    payload.tools = request.tools
  else
    payload.tools = {}
  end
  if request.tool_choice then payload.tool_choice = request.tool_choice end
  if type(request.response_format) == "table" then payload.response_format = request.response_format end
  local headers = opencode_headers(api_key, ses, msg)
  local _, stream_err = client:stream({
    method = "POST",
    url = BASE_URL .. endpoint,
    headers = headers,
    body = json.encode(payload),
    proxy_url = proxy_url,
    on_response = function(r)
      if r.status == 200 then return end
      local action, terr = map_upstream(ctx, r, cred_id)
      if action == "done" then st.fatal = true end
      if action == "cred" then st.next_cred = true end
      return terr
    end,
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
      st.done = true
    end,
  })
  return stream_err
end

llm_router.register("opencode-zen", {
  icon = "https://opencode.ai/favicon.ico",

  proxy_schema = {},

  credential_schema = {
    {
      type = "section",
      title = "OpenCode Zen",
      content = {
        { type = "banner", variant = "info", text = "Zen API key required. Paid models only." },
        { type = "secret", name = "api_key", label = "API Key", required = true },
        { type = "button", text = "Save", form_action = "submit" },
      },
    },
  },

  validate_credentials = function(data)
    local key = data.api_key
    if key == nil or key == "" then
      return false, { message = "api_key is required", code = "invalid_request_error" }
    end
    if #key < 20 then return false, { message = "api_key: minimum 20 characters", code = "invalid_request_error" } end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    local key = ""
    if #creds > 0 then key = api_key_of(creds[1].data) end
    if key == "" then return with_limits(FALLBACK_MODELS) end
    local client = llm_router.http_client({})
    local ses, msg = mint_ids()
    local headers = opencode_headers(key, ses, msg)
    local resp, err = client:request({
      method = "GET",
      url = BASE_URL .. "/models",
      headers = headers,
    })
    if err then
      -- Upstream listing failed: fall back to the known model set.
      return with_limits(FALLBACK_MODELS)
    end
    if resp.status ~= 200 then return with_limits(FALLBACK_MODELS) end
    local ok, parsed = pcall(json.decode, resp.body)
    if not ok or not parsed or type(parsed.data) ~= "table" then return with_limits(FALLBACK_MODELS) end
    local infos = {}
    for _, m in ipairs(parsed.data) do
      if type(m.id) == "string" and m.id ~= "" then table.insert(infos, { name = m.id, display_name = m.id }) end
    end
    if #infos == 0 then return with_limits(FALLBACK_MODELS) end
    return with_limits(infos)
  end,

  -- Account liveness: the keyed /models listing answers 200 for live keys
  -- and 401 for dead ones. Only 401 verdicts unhealthy; every other
  -- failure is unknown and changes nothing.
  check_health = function(ctx, credential)
    local key = api_key_of(credential.data)
    if key == "" then return { status = "unhealthy", message = "api_key is required" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local ses, msg = mint_ids()
    local resp, err = client:request({
      method = "GET",
      url = BASE_URL .. "/models",
      headers = opencode_headers(key, ses, msg),
    })
    if err then
      local message = "request failed"
      if type(err) == "table" and type(err.message) == "string" then message = err.message end
      return { status = "unknown", message = message }
    end
    if resp.status == 200 then return { status = "healthy" } end
    if resp.status == 401 then return { status = "unhealthy", message = "api key rejected" } end
    return { status = "unknown", message = "status " .. tostring(resp.status) }
  end,

  complete = function(ctx, request)
    local model = request.model_name
    local endpoint = endpoint_for_model(model)
    local client = llm_router.http_client({})
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      local api_key = api_key_of(cred.data)
      if api_key == "" then
        last_err = { message = "api_key is required", code = "invalid_request_error", status = 400 }
      else
        for _, px in ipairs(pick_proxies(ctx, 3)) do
          local ses, msg = mint_ids()
          local resp, action, err
          if endpoint == "/responses" then
            resp, action, err = do_responses_unary(client, request, model, api_key, ses, msg, px.url, cred.id)
          else
            resp, action, err = do_chat_unary(client, request, model, endpoint, api_key, ses, msg, px.url, cred.id)
          end
          if resp then return resp end
          if action == "done" then return nil, err end
          last_err = err
          if action == "cred" then break end
        end
      end
    end
    return nil, last_err or { message = "all zen credentials exhausted", code = "server_error" }
  end,

  complete_stream = function(ctx, request, emit)
    local model = request.model_name
    local endpoint = endpoint_for_model(model)
    local client = llm_router.http_client({})
    local full_model = request.model
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      local api_key = api_key_of(cred.data)
      if api_key == "" then
        last_err = { message = "api_key is required", code = "invalid_request_error", status = 400 }
      else
        for _, px in ipairs(pick_proxies(ctx, 3)) do
          local ses, msg = mint_ids()
          local st = { done = false, fatal = false, next_cred = false }
          local stream_err
          if endpoint == "/responses" then
            stream_err = do_responses_stream(client, request, model, api_key, ses, msg, px.url, cred.id, emit, st)
          else
            stream_err =
              do_chat_stream(client, request, model, endpoint, api_key, ses, msg, px.url, cred.id, full_model, emit, st)
          end
          if stream_err == nil then return end
          last_err = stream_err
          if st.fatal then return nil, last_err end
          if st.next_cred then break end
          -- A stream that emitted already belongs to that attempt.
          if st.done then return nil, last_err end
        end
      end
    end
    return nil, last_err or { message = "all zen credentials exhausted", code = "server_error" }
  end,
})
