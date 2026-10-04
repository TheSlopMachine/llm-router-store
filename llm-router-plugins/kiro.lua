--- @plugin Kiro AI
--- @author TheSlopMachine
--- @version 4.0.3
--- @router_version 0.7.0
--- @description AWS Kiro models via device login (OAuth2 with proactive refresh)
--- @allow_host codewhisperer.us-east-1.amazonaws.com
--- @allow_host oidc.us-east-1.amazonaws.com
--- @allow_host q.us-east-1.amazonaws.com
--- @allow_host q.eu-central-1.amazonaws.com

local DEFAULT_REGION = "us-east-1"
local BUILDER_START_URL = "https://view.awsapps.com/start"
local ISSUER_URL = "https://identitycenter.amazonaws.com/ssoins-722374e8c3c8e6c6"
local CONVERSATION_NS = "34f7193f-561d-4050-bc84-9547d953d6bf"

local function oidc_url(region, path) return "https://oidc." .. region .. ".amazonaws.com/" .. path end

local function region_of(credential, provider_config)
  if credential and credential.data and credential.data.region and credential.data.region ~= "" then
    return credential.data.region
  end
  if provider_config and provider_config.region and provider_config.region ~= "" then return provider_config.region end
  return DEFAULT_REGION
end

-- ── Runtime region + model discovery ──

-- Q Developer profiles live in us-east-1/eu-central-1 only. A stored IdC
-- region is a token region, not a runtime region. The profileArn region
-- wins when present; hosts come from a fixed list, never raw input.
local PROFILE_REGIONS = { "us-east-1", "eu-central-1" }

local function valid_region(s) return type(s) == "string" and s:lower():match("^[a-z][a-z]-[a-z]+-%d+$") ~= nil end

local function region_from_arn(arn)
  if type(arn) ~= "string" then return nil end
  local region = arn:lower():match("^arn:aws:codewhisperer:([a-z0-9-]+):")
  if region and valid_region(region) then return region end
  return nil
end

local function runtime_region(credential, provider_config)
  local data = credential and credential.data or {}
  local arn_region = region_from_arn(data.profile_arn) or region_from_arn(data.profileArn)
  if arn_region then return arn_region end
  local stored = region_of(credential, provider_config):lower()
  for _, region in ipairs(PROFILE_REGIONS) do
    if stored == region then return stored end
  end
  return DEFAULT_REGION
end

local function runtime_host(region)
  if region == "us-east-1" then return "https://codewhisperer.us-east-1.amazonaws.com" end
  return "https://q." .. region .. ".amazonaws.com"
end

local function generate_url(credential, provider_config)
  return runtime_host(runtime_region(credential, provider_config)) .. "/generateAssistantResponse"
end

local function discovery_endpoints(region)
  local urls = { "https://q." .. region .. ".amazonaws.com/ListAvailableModels" }
  if region ~= "us-east-1" then table.insert(urls, "https://q.us-east-1.amazonaws.com/ListAvailableModels") end
  return urls
end

local function parse_model_list(body)
  local ok, parsed = pcall(json.decode, body)
  if not ok or type(parsed) ~= "table" then return nil end
  local items = parsed.models
  if type(items) ~= "table" then items = parsed.availableModels end
  if type(items) ~= "table" then return nil end
  local out, seen = {}, {}
  for _, item in ipairs(items) do
    if type(item) == "table" then
      local id = item.modelId
      if type(id) ~= "string" or id == "" then id = item.id end
      if type(id) == "string" and id ~= "" and not seen[id] then
        seen[id] = true
        local name = item.modelName
        if type(name) ~= "string" or name == "" then name = item.name end
        if type(name) ~= "string" or name == "" then name = id end
        local context = 200000
        if type(item.tokenLimits) == "table" then
          local max_input = tonumber(item.tokenLimits.maxInputTokens)
          if max_input and max_input > 0 then context = math.floor(max_input) end
        end
        table.insert(out, {
          name = id,
          display_name = name,
          context_window = context,
          max_tokens = 32000,
          supported_parameters = { "tools" },
          input_modalities = { "text" },
          output_modalities = { "text" },
          endpoints = { "chat/completions" },
        })
      end
    end
  end
  if #out == 0 then return nil end
  return out
end

local function fetch_model_list(client, url, token, profile_arn)
  local target = url .. "?origin=AI_EDITOR"
  if type(profile_arn) == "string" and profile_arn ~= "" then target = target .. "&profileArn=" .. profile_arn end
  local resp, err = client:request({
    method = "GET",
    url = target,
    headers = {
      ["Authorization"] = "Bearer " .. token,
      ["Accept"] = "application/json",
      ["User-Agent"] = "AWS-SDK-JS/3.0.0 kiro-ide/1.0.0",
    },
  })
  if err then return nil end
  if resp.status ~= 200 then return nil end
  return parse_model_list(resp.body)
end

-- ── Thinking support ──

-- Only these models accept Kiro thinking controls. Adaptive envelope
-- (output_config + thinking type) for Claude; native reasoning.effort
-- for GPT-5.6. Anything else with a -thinking suffix fails closed:
-- Kiro 400s unknown thinking requests.
local ADAPTIVE_THINKING_MODELS = { ["claude-opus-5"] = true, ["claude-sonnet-5"] = true }
local NATIVE_REASONING_MODELS = { ["gpt-5.6-sol"] = true, ["gpt-5.6-terra"] = true, ["gpt-5.6-luna"] = true }
local EFFORT_BUDGETS = { low = 8000, medium = 16000, high = 32000, xhigh = 64000, max = 120000 }

local function resolve_effort(request)
  local effort = ""
  if request and type(request.reasoning_effort) == "string" then effort = request.reasoning_effort:lower() end
  if effort == "minimal" then effort = "low" end
  if effort == "" or effort == "none" then return "" end
  if not EFFORT_BUDGETS[effort] then return "" end
  return effort
end

-- Splits a requested model into its upstream id plus thinking flag. A
-- -thinking suffix enables thinking; an explicit effort on an allowlisted
-- model does the same without the suffix. Anything else thinking-flavored
-- returns nil plus an invalid_request table.
local function resolve_model(model, effort)
  local upstream, thinking = model, false
  if type(model) == "string" and model:sub(-9) == "-thinking" then
    thinking = true
    upstream = model:sub(1, -10)
  elseif effort ~= "" and (ADAPTIVE_THINKING_MODELS[model] or NATIVE_REASONING_MODELS[model]) then
    thinking = true
    upstream = model
  end
  if thinking and not ADAPTIVE_THINKING_MODELS[upstream] and not NATIVE_REASONING_MODELS[upstream] then
    return nil,
      false,
      {
        message = "model " .. tostring(model) .. " does not support thinking on kiro",
        code = "invalid_request_error",
        status = 400,
      }
  end
  return upstream, thinking, nil
end

-- Appends -thinking variants for allowlisted models to a discovered list.
local function with_thinking_variants(models)
  local out = {}
  for _, m in ipairs(models) do
    table.insert(out, m)
    if ADAPTIVE_THINKING_MODELS[m.name] or NATIVE_REASONING_MODELS[m.name] then
      table.insert(out, {
        name = m.name .. "-thinking",
        display_name = m.display_name .. " (Thinking)",
        context_window = m.context_window,
        max_tokens = m.max_tokens,
        supported_parameters = { "tools" },
        input_modalities = { "text" },
        output_modalities = { "text" },
        endpoints = { "chat/completions" },
        reasoning = { default_enabled = true, supported_efforts = { "max", "xhigh", "high", "medium", "low" } },
      })
    end
  end
  return out
end

-- Reasoning trace out of a reasoningContentEvent payload: object form
-- ({text} or {Text}), plain string form, or flat {text}.
local function reasoning_text(p)
  local rt = p.reasoningText
  if type(rt) == "table" then
    if type(rt.text) == "string" and rt.text ~= "" then return rt.text end
    if type(rt.Text) == "string" and rt.Text ~= "" then return rt.Text end
    return ""
  end
  if type(rt) == "string" and rt ~= "" then return rt end
  if type(p.text) == "string" then return p.text end
  return ""
end

local THINK_OPEN, THINK_CLOSE = "<thinking>", "</thinking>"

-- Stream-safe splitter: routes one content slice into content vs reasoning
-- buckets by <thinking> state. A tag split across frames is held in
-- st.pending (</thinking> is the longest at 11 chars) and completed on
-- the next slice.
local function split_thinking(st, raw, content_arr, reasoning_arr)
  local text = (st.pending or "") .. (raw or "")
  st.pending = ""
  local function emit(s)
    if s == "" then return end
    if st.thinkingMode then
      table.insert(reasoning_arr, s)
    else
      table.insert(content_arr, s)
    end
  end
  while #text > 0 do
    local tag = st.thinkingMode and THINK_CLOSE or THINK_OPEN
    local s, e = text:find(tag, 1, true)
    if not s then
      local hold = #text + 1
      local from = math.max(1, #text - 10)
      for i = from, #text do
        if tag:sub(1, #text - i + 1) == text:sub(i) then
          hold = i
          break
        end
      end
      emit(text:sub(1, hold - 1))
      st.pending = text:sub(hold)
      return
    end
    emit(text:sub(1, s - 1))
    st.thinkingMode = not st.thinkingMode
    text = text:sub(e + 1)
  end
end

-- ── Binary AWS event-stream framing ──

local function u32be(s, pos)
  local a, b, c, d = string.byte(s, pos, pos + 3)
  return ((a * 256 + b) * 256 + c) * 256 + d
end

-- Split a buffer into complete frames, returning frames + remainder.
local function split_frames(buf)
  local frames = {}
  local pos = 1
  while #buf - pos + 1 >= 16 do
    local total = u32be(buf, pos)
    if total < 16 or pos + total - 1 > #buf then break end
    table.insert(frames, buf:sub(pos, pos + total - 1))
    pos = pos + total
  end
  return frames, buf:sub(pos)
end

local function parse_frame(frame)
  local headers_len = u32be(frame, 5)
  local headers = {}
  local off = 13
  local header_end = 12 + headers_len
  while off < header_end do
    local name_len = string.byte(frame, off)
    off = off + 1
    local name = frame:sub(off, off + name_len - 1)
    off = off + name_len
    local vtype = string.byte(frame, off)
    off = off + 1
    if vtype ~= 7 then return nil end
    local vlen = string.byte(frame, off) * 256 + string.byte(frame, off + 1)
    off = off + 2
    headers[name] = frame:sub(off, off + vlen - 1)
    off = off + vlen
  end
  local payload = {}
  local payload_end = #frame - 4
  if payload_end > header_end then
    local raw = frame:sub(header_end + 1, payload_end):match("^%s*(.-)%s*$")
    if raw ~= "" then
      local ok, parsed = pcall(json.decode, raw)
      if ok and type(parsed) == "table" then
        payload = parsed
      else
        payload = { raw = raw }
      end
    end
  end
  return { headers = headers, payload = payload }
end

local function new_aggregate(thinking)
  return {
    content = {},
    reasoning = {},
    prompt_tokens = 0,
    completion_tokens = 0,
    total_tokens = 0,
    finish_reason = "",
    tool_calls = {},
    builders = {},
    order = {},
    split = thinking == true,
    think_state = { thinkingMode = false, pending = "" },
  }
end

local function flush_builder(state, id)
  local b = state.builders[id]
  if b == nil or b.name == nil or b.name == "" then return end
  local args = b.args:match("^%s*(.-)%s*$")
  if args == "" then args = "{}" end
  table.insert(state.tool_calls, { id = id, type = "function", ["function"] = { name = b.name, arguments = args } })
  state.finish_reason = "tool_calls"
  state.builders[id] = nil
end

local function consume_event(state, frame)
  local etype = frame.headers[":event-type"]
  local p = frame.payload
  if etype == "assistantResponseEvent" or etype == "codeEvent" then
    local content = p.content
    if type(content) == "string" and content ~= "" then
      if state.split then
        split_thinking(state.think_state, content, state.content, state.reasoning)
      else
        table.insert(state.content, content)
      end
    end
  elseif etype == "reasoningContentEvent" then
    local text = reasoning_text(p)
    if text ~= "" then table.insert(state.reasoning, text) end
  elseif etype == "metricsEvent" then
    local m = p.metricsEvent or p
    state.prompt_tokens = m.inputTokens or 0
    state.completion_tokens = m.outputTokens or 0
    state.total_tokens = state.prompt_tokens + state.completion_tokens
  elseif etype == "toolUseEvent" then
    local id = p.toolUseId or ""
    if id == "" then id = "call_" .. tostring(os.time()) end
    local b = state.builders[id]
    if b == nil then
      b = { name = "", args = "" }
      state.builders[id] = b
      table.insert(state.order, id)
    end
    if type(p.name) == "string" and p.name ~= "" then b.name = p.name end
    local is_string_input = false
    if p.input ~= nil then
      if type(p.input) == "string" then
        b.args = b.args .. p.input
        is_string_input = true
      else
        b.args = json.encode(p.input)
      end
    end
    local should_emit = false
    if p.stop == true then
      should_emit = true
    elseif not is_string_input and p.input ~= nil and b.name ~= "" then
      should_emit = true
    end
    if should_emit then flush_builder(state, id) end
  elseif etype == "messageStopEvent" then
    if state.finish_reason == "" then state.finish_reason = "stop" end
  end
end

local function aggregate_response(body, thinking)
  local state = new_aggregate(thinking)
  local frames = split_frames(body)
  for _, f in ipairs(frames) do
    local frame = parse_frame(f)
    if frame then consume_event(state, frame) end
  end
  if state.think_state.pending ~= "" then
    if state.think_state.thinkingMode then
      table.insert(state.reasoning, state.think_state.pending)
    else
      table.insert(state.content, state.think_state.pending)
    end
    state.think_state.pending = ""
  end
  for _, id in ipairs(state.order) do
    if state.builders[id] and state.builders[id].name ~= "" then flush_builder(state, id) end
  end
  return state
end

-- ── Request transform ──

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

local function build_tool_specs(tools)
  local specs = {}
  for _, t in ipairs(tools or {}) do
    local name = t.name or ""
    local desc = t.description or ""
    local schema = t.parameters
    if t["function"] then
      if t["function"].name and t["function"].name ~= "" then name = t["function"].name end
      if t["function"].description and t["function"].description ~= "" then desc = t["function"].description end
      if t["function"].parameters then schema = t["function"].parameters end
    end
    if name ~= "" then
      if desc == "" then desc = "Tool: " .. name end
      if schema == nil then schema = { type = "object", properties = {} } end
      table.insert(
        specs,
        {
          toolSpecification = {
            name = name,
            description = desc,
            inputSchema = { json = schema },
            strict = true,
          },
        }
      )
    end
  end
  return specs
end

local function convert_messages(messages, tools, model)
  local history = {}
  local current_role = nil
  local user_parts = {}
  local assistant_parts = {}
  local tool_results = {}

  local function flush_user()
    local content = table.concat(user_parts, "\n\n"):match("^%s*(.-)%s*$")
    if content == "" and #tool_results == 0 then
      user_parts = {}
      return
    end
    local msg = { content = content, modelId = model }
    if msg.content == "" then msg.content = "continue" end
    if #tool_results > 0 then msg.userInputMessageContext = { toolResults = tool_results } end
    if #tools > 0 and #history == 0 then
      msg.userInputMessageContext = msg.userInputMessageContext or {}
      msg.userInputMessageContext.tools = build_tool_specs(tools)
    end
    table.insert(history, { userInputMessage = msg })
    user_parts = {}
    tool_results = {}
  end

  local function flush_assistant()
    local content = table.concat(assistant_parts, "\n\n"):match("^%s*(.-)%s*$")
    if content == "" then content = "..." end
    table.insert(history, { assistantResponseMessage = { content = content } })
    assistant_parts = {}
  end

  for _, m in ipairs(messages or {}) do
    local role = "user"
    if m.role == "assistant" then role = "assistant" end
    if current_role and current_role ~= role then
      if current_role == "user" then
        flush_user()
      else
        flush_assistant()
      end
    end
    current_role = role
    if role == "assistant" then
      local text = message_text(m):match("^%s*(.-)%s*$")
      if text ~= "" then table.insert(assistant_parts, text) end
      local uses = {}
      for _, tc in ipairs(m.tool_calls or {}) do
        local name = tc["function"] and tc["function"].name or ""
        if name ~= "" then
          local args = tc["function"].arguments or "{}"
          local ok, parsed = pcall(json.decode, args)
          if not ok or type(parsed) ~= "table" then parsed = {} end
          table.insert(uses, { toolUseId = tc.id or "", name = name, input = parsed })
        end
      end
      if #uses > 0 then
        flush_assistant()
        history[#history].assistantResponseMessage.toolUses = uses
        current_role = nil
      end
    else
      if m.role == "tool" then
        table.insert(
          tool_results,
          { toolUseId = m.tool_call_id or "", status = "success", content = { { text = message_text(m) } } }
        )
      else
        local text = message_text(m):match("^%s*(.-)%s*$")
        if text ~= "" then table.insert(user_parts, text) end
      end
    end
  end
  if current_role == "user" then
    flush_user()
  elseif current_role == "assistant" then
    flush_assistant()
  end

  local current = nil
  if #history > 0 and history[#history].userInputMessage then
    current = history[#history].userInputMessage
    table.remove(history)
  end
  if current == nil or current.content == "" then current = { content = "continue", modelId = model } end
  if current.modelId == nil or current.modelId == "" then current.modelId = model end
  if #tools > 0 then
    current.userInputMessageContext = current.userInputMessageContext or {}
    if current.userInputMessageContext.tools == nil then
      current.userInputMessageContext.tools = build_tool_specs(tools)
    end
  end
  for _, entry in ipairs(history) do
    local e = entry.userInputMessage
    if e and e.userInputMessageContext then
      e.userInputMessageContext.tools = nil
      if e.userInputMessageContext.toolResults == nil then e.userInputMessageContext = nil end
    end
    if e and (e.modelId == nil or e.modelId == "") then e.modelId = model end
  end
  return history, current
end

local function conversation_id(history, current_content)
  local seed = current_content or ""
  if
    #history > 0
    and history[1].userInputMessage
    and type(history[1].userInputMessage.content) == "string"
    and history[1].userInputMessage.content ~= ""
  then
    seed = history[1].userInputMessage.content
  end
  if #seed > 4000 then seed = seed:sub(1, 4000) end
  return llm_router.uuid_v5(CONVERSATION_NS, seed)
end

local function build_payload(request, model, thinking, effort)
  local history, current = convert_messages(request.messages, request.tools or {}, model)
  if current.content == "" then current.content = "continue" end
  current.content = "[Context: Current time is " .. os.date("!%Y-%m-%dT%H:%M:%SZ") .. "]\n\n" .. current.content
  if thinking then
    if effort == "" then effort = "high" end
    current.content = "<thinking_mode>enabled</thinking_mode>"
      .. "<max_thinking_length>"
      .. tostring(EFFORT_BUDGETS[effort] or EFFORT_BUDGETS.high)
      .. "</max_thinking_length>"
      .. "\n\n"
      .. current.content
  end
  current.origin = "AI_EDITOR"
  local state = {
    chatTriggerType = "MANUAL",
    conversationId = conversation_id(history, current.content),
    currentMessage = { userInputMessage = current },
  }
  if #history > 0 then state.history = history end
  local payload = { conversationState = state }
  if
    (request.max_tokens and request.max_tokens > 0)
    or ((request.temperature and request.temperature > 0) and not thinking)
    or ((request.top_p and request.top_p > 0) and not thinking)
  then
    payload.inferenceConfig = {}
    if request.max_tokens and request.max_tokens > 0 then payload.inferenceConfig.maxTokens = request.max_tokens end
    if not thinking then
      if request.temperature and request.temperature > 0 then
        payload.inferenceConfig.temperature = request.temperature
      end
      if request.top_p and request.top_p > 0 then payload.inferenceConfig.topP = request.top_p end
    end
  end
  if thinking then
    if NATIVE_REASONING_MODELS[model] then
      payload.additionalModelRequestFields = { reasoning = { effort = effort } }
    else
      payload.additionalModelRequestFields = {
        output_config = { effort = effort },
        thinking = { type = "adaptive", display = "summarized" },
      }
    end
  end
  return payload
end

local function generate_headers(credential)
  local token = ""
  if credential and credential.data then token = credential.data.access_token or "" end
  return {
    ["Authorization"] = "Bearer " .. token,
    ["Content-Type"] = "application/json",
    ["Accept"] = "application/vnd.amazon.eventstream",
    ["X-Amz-Target"] = "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
    ["User-Agent"] = "AWS-SDK-JS/3.0.0 kiro-ide/1.0.0",
    ["X-Amz-User-Agent"] = "aws-sdk-js/3.0.0 kiro-ide/1.0.0",
    ["Amz-Sdk-Request"] = "attempt=1; max=3",
    ["Amz-Sdk-Invocation-Id"] = llm_router.random_hex(16),
    ["x-amzn-bedrock-cache-control"] = "enable",
    ["anthropic-beta"] = "prompt-caching-2024-07-31",
  }
end

-- ── OAuth device flow (auth wizard) ──

local function method_page(error_text)
  local nodes = {}
  if error_text and error_text ~= "" then
    table.insert(nodes, { type = "banner", variant = "error", text = error_text })
  end
  table.insert(
    nodes,
    {
      type = "section",
      title = "Sign in to Kiro",
      subtitle = "Use the same Kiro account as in the IDE.",
      content = {
        {
          type = "select",
          name = "device_method",
          label = "Method",
          options = { "builder-id", "idc" },
          option_labels = { ["builder-id"] = "AWS Builder ID", ["idc"] = "IAM Identity Center" },
          value = "builder-id",
        },
        { type = "button", text = "Continue", form_action = "pick_method" },
      },
    }
  )
  return { render = nodes }
end

local function region_page(error_text, region, start_url)
  local nodes = {}
  if error_text and error_text ~= "" then
    table.insert(nodes, { type = "banner", variant = "error", text = error_text })
  end
  table.insert(
    nodes,
    {
      type = "section",
      title = "Sign in to Kiro",
      subtitle = "Enter your IAM Identity Center details.",
      content = {
        {
          type = "grid",
          columns = 2,
          content = {
            { type = "input", name = "region", label = "Region", value = region or DEFAULT_REGION },
            { type = "input", name = "start_url", label = "Start URL", value = start_url or BUILDER_START_URL },
          },
        },
        { type = "button", text = "Start Device Login", form_action = "start_device" },
        { type = "button", text = "Back", form_action = "restart" },
      },
    }
  )
  return { render = nodes }
end

local function builder_page()
  local nodes = {
    {
      type = "section",
      title = "Sign in to Kiro",
      subtitle = "Uses the Kiro default start URL in us-east-1. No input needed.",
      content = {
        { type = "button", text = "Start Device Login", form_action = "start_device" },
        { type = "button", text = "Back", form_action = "restart" },
      },
    },
  }
  return { render = nodes }
end

local function start_page(error_text, region, start_url, method)
  if method == "idc" then return region_page(error_text, region, start_url) end
  return method_page(error_text)
end

local function device_page(message_text, state)
  local nodes = {}
  if message_text and message_text ~= "" then
    table.insert(nodes, { type = "banner", variant = "info", text = message_text })
  end
  table.insert(
    nodes,
    {
      type = "section",
      title = "Complete device login",
      subtitle = "Open the verification page, then enter the code below.",
      content = {
        { type = "code", text = state.user_code or "", label = "Device code" },
        {
          type = "link",
          text = "Open verification page",
          url = state.verification_uri_complete or state.verification_uri or "",
        },
        { type = "button", text = "Check Authorization", form_action = "poll_device" },
        { type = "button", text = "Start Over", form_action = "restart" },
      },
    }
  )
  return { render = nodes }
end

local function flow_scope(flow_id) return "auth_flow:" .. flow_id end

local function token_of(data)
  if type(data) ~= "table" then return "" end
  return data.access_token or ""
end

-- Dead-key bench: shared disable when automation is on (visible in the
-- dashboard, global to every path), unified cooldown park otherwise.
local function bench(ctx, cred_id, reason, wait_secs)
  if wait_secs == nil or wait_secs < 1 then wait_secs = 300 end
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
  local proxies = llm_router.proxies.query({ pool = pool, limit = limit or 3 })
  if #proxies == 0 then
    -- Empty pool degrades to one direct attempt, never to silence.
    return { {} }
  end
  return proxies
end

-- Kiro status mapping: 429 quota wording parks the OAuth account, other
-- outcomes move on. Returns "proxy" (next exit, same credential), "cred"
-- (next credential) or "done" (terminal, return at once).
local function map_upstream(ctx, resp, cred_id)
  if resp.status == 401 then
    bench(ctx, cred_id, "kiro rejected the access token", 300)
    return "cred", { message = "kiro rejected the access token", code = "authentication_error", status = 401 }
  end
  if resp.status == 429 then
    llm_router.credentials.park(cred_id, retry_after_secs(resp), "kiro rate limited")
    local lower = string.lower(tostring(resp.body or ""))
    if string.find(lower, "quota", 1, true) then
      return "cred", { message = "kiro quota exhausted", code = "insufficient_quota", status = 429 }
    end
    return "cred", { message = "kiro rate limited", code = "rate_limit", status = 429 }
  end
  if resp.status == 400 or resp.status == 404 then
    return "done",
      {
        message = "kiro rejected the request with status " .. tostring(resp.status),
        code = "invalid_request_error",
        status = resp.status,
      }
  end
  return "proxy",
    { message = "kiro returned status " .. tostring(resp.status), code = "server_error", status = resp.status }
end

-- Expiry check moved out of needs_refresh: true when the token needs a
-- refresh attempt. Pasted tokens without expiry metadata only qualify when
-- the device flow stored OIDC client credentials.
local function is_token_stale(data)
  data = data or {}
  if not data.refresh_token or data.refresh_token == "" then return false end
  if not data.access_token or data.access_token == "" then return true end
  if not data.expires_at or data.expires_at == "" then return (data.client_id or "") ~= "" end
  local now = os.time()
  local exp = nil
  local y, mo, d, h, mi, s = data.expires_at:match("^(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)")
  if y then
    exp = os.time({
      year = tonumber(y),
      month = tonumber(mo),
      day = tonumber(d),
      hour = tonumber(h),
      min = tonumber(mi),
      sec = tonumber(s),
    })
  else
    exp = tonumber(data.expires_at)
  end
  if exp == nil then return false end
  return (exp - now) <= 300
end

-- OIDC refresh moved out of refresh_credential: returns the merged data
-- map on success, nil on any failure (the next tick retries).
local function do_refresh(data)
  local region = data.region
  if not region or region == "" then region = DEFAULT_REGION end
  local client = llm_router.http_client({})
  local resp, err = client:request({
    method = "POST",
    url = oidc_url(region, "token"),
    headers = { ["Content-Type"] = "application/json", ["Accept"] = "application/json" },
    body = json.encode({
      clientId = data.client_id or "",
      clientSecret = data.client_secret or "",
      refreshToken = data.refresh_token or "",
      grantType = "refresh_token",
    }),
  })
  if err then return nil end
  if resp.status ~= 200 then return nil end
  local out = json.decode(resp.body)
  if not out.accessToken or out.accessToken == "" then return nil end
  local merged = {}
  for k, v in pairs(data) do
    merged[k] = v
  end
  merged.access_token = out.accessToken
  if out.refreshToken and out.refreshToken ~= "" then merged.refresh_token = out.refreshToken end
  if out.expiresIn and out.expiresIn > 0 then
    merged.expires_at = os.date("!%Y-%m-%dT%H:%M:%SZ", os.time() + out.expiresIn)
  end
  return merged
end

-- Shared unary completion assembly: aggregated Kiro response → OpenAI.
local function assemble_completion(body, thinking, full_model)
  local state = aggregate_response(body, thinking)
  local text = table.concat(state.content, "")
  local reasoning = table.concat(state.reasoning, "")
  local finish = state.finish_reason
  if finish == "" then finish = "stop" end
  if #state.tool_calls > 0 then finish = "tool_calls" end
  local total = state.total_tokens
  if total == 0 then
    local completion = math.max(1, math.floor(#body / 4))
    total = state.prompt_tokens + completion
    state.completion_tokens = completion
  end
  local message = { role = "assistant", content = text, tool_calls = state.tool_calls }
  if reasoning ~= "" then message.reasoning_content = reasoning end
  return {
    id = "kiro-" .. tostring(os.time()),
    object = "chat.completion",
    created = os.time(),
    model = full_model,
    choices = {
      { index = 0, message = message, finish_reason = finish },
    },
    usage = {
      prompt_tokens = state.prompt_tokens,
      completion_tokens = state.completion_tokens,
      total_tokens = total,
    },
  }
end

llm_router.register("kiro", {
  icon = "https://kiro.dev/favicon.ico",

  proxy_schema = {},

  config_schema = {
    { type = "select", name = "region", label = "Region", options = { "us-east-1" } },
  },

  credential_schema = {
    {
      type = "section",
      title = "Manual token entry",
      subtitle = "Paste tokens from a previous login, or use device login instead.",
      content = {
        { type = "secret", name = "access_token", label = "Access Token" },
        { type = "secret", name = "refresh_token", label = "Refresh Token" },
        { type = "button", text = "Save", form_action = "submit" },
      },
    },
  },

  jobs = {
    refresh = {
      interval_seconds = 300,
      run_on_startup = true,
      timeout_ms = 30000,
      run = function(ctx)
        for _, c in ipairs(llm_router.credentials.list()) do
          local data = c.data or {}
          if is_token_stale(data) then
            local merged, err = do_refresh(data)
            if merged then llm_router.credentials.update(c.id, merged) end
          end
        end
        return true
      end,
    },
  },

  validate_credentials = function(data)
    local access = data.access_token or ""
    local refresh = data.refresh_token or ""
    if access == "" and refresh == "" then
      return false, { message = "either access_token or refresh_token is required", code = "invalid_request_error" }
    end
    if access ~= "" and #access < 20 then
      return false, { message = "access_token appears invalid (too short)", code = "invalid_request_error" }
    end
    if refresh ~= "" and #refresh < 20 then
      return false, { message = "refresh_token appears invalid (too short)", code = "invalid_request_error" }
    end
    return true
  end,

  get_model_infos = function(ctx)
    local creds = llm_router.credentials.list()
    if #creds == 0 then
      return nil, { message = "no kiro credentials configured", code = "invalid_request_error", status = 400 }
    end
    local first = creds[1]
    local token = token_of(first.data)
    if token == "" then
      return nil,
        {
          message = "kiro access token is required for model discovery",
          code = "authentication_error",
          status = 401,
        }
    end
    local data = first.data or {}
    local region = runtime_region(first, ctx.provider_config)
    local client = llm_router.http_client({})
    for _, url in ipairs(discovery_endpoints(region)) do
      local models = fetch_model_list(client, url, token)
      if models then return with_thinking_variants(models) end
    end
    -- Desktop-style accounts serve the catalog under their profile ARN.
    -- Builder ID must not send it (yields 403), so this stays a retry.
    local arn = data.profile_arn
    if type(arn) ~= "string" or arn == "" then arn = data.profileArn end
    if type(arn) == "string" and arn ~= "" then
      local models = fetch_model_list(client, discovery_endpoints(region)[1], token, arn)
      if models then return with_thinking_variants(models) end
    end
    return nil, { message = "kiro model discovery failed on every region", code = "server_error", status = 502 }
  end,

  -- Account liveness: the ListAvailableModels catalog answers 200 for live
  -- tokens and 401 for dead ones. Only 401 verdicts unhealthy; region and
  -- profile-ARN mismatches surface as 403s, which stay unknown and change
  -- nothing. Single primary-region attempt, no fallback replication.
  check_health = function(ctx, credential, provider_config)
    local data = (credential and credential.data) or {}
    local token = data.access_token or ""
    if token == "" then return { status = "unhealthy", message = "access_token is required" } end
    local region = runtime_region(credential, provider_config)
    local target = discovery_endpoints(region)[1] .. "?origin=AI_EDITOR"
    local arn = data.profile_arn
    if type(arn) ~= "string" or arn == "" then arn = data.profileArn end
    if type(arn) == "string" and arn ~= "" then target = target .. "&profileArn=" .. arn end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local resp, err = client:request({
      method = "GET",
      url = target,
      headers = {
        ["Authorization"] = "Bearer " .. token,
        ["Accept"] = "application/json",
        ["User-Agent"] = "AWS-SDK-JS/3.0.0 kiro-ide/1.0.0",
      },
    })
    if err then
      local message = "request failed"
      if type(err) == "table" and type(err.message) == "string" then message = err.message end
      return { status = "unknown", message = message }
    end
    if resp.status == 200 then return { status = "healthy" } end
    if resp.status == 401 then return { status = "unhealthy", message = "access token rejected" } end
    return { status = "unknown", message = "status " .. tostring(resp.status) }
  end,

  auth_initiate = function(ctx) return start_page("", DEFAULT_REGION, BUILDER_START_URL, "builder-id") end,

  auth_step = function(ctx, input)
    local action = input.action or ""
    local values = input.values or {}
    local scope = flow_scope(ctx.flow_id or "")

    if action == "restart" or action == "cancel" then
      llm_router.storage.delete(scope, "device")
      llm_router.storage.delete(scope, "method")
      return start_page("", DEFAULT_REGION, BUILDER_START_URL, "builder-id")
    end

    if action == "pick_method" then
      local method = values.device_method or "builder-id"
      if method == "" then method = "builder-id" end
      if method ~= "builder-id" and method ~= "idc" then
        return method_page("Choose a device login method to continue.")
      end
      llm_router.storage.set(scope, "method", method)
      if method == "builder-id" then return builder_page() end
      return region_page("", DEFAULT_REGION, BUILDER_START_URL)
    end

    if action == "start_device" then
      local method = llm_router.storage.get(scope, "method") or "builder-id"
      if method == "" then method = "builder-id" end
      local region = values.region or DEFAULT_REGION
      if region == "" then region = DEFAULT_REGION end
      local start_url = values.start_url or BUILDER_START_URL
      if start_url == "" then start_url = BUILDER_START_URL end
      if method ~= "builder-id" and method ~= "idc" then
        return start_page("Unsupported device login method.", region, start_url, method)
      end
      if method == "builder-id" then start_url = BUILDER_START_URL end

      local client = llm_router.http_client({})
      local reg_resp, reg_err = client:request({
        method = "POST",
        url = oidc_url(region, "client/register"),
        headers = { ["Content-Type"] = "application/json" },
        body = json.encode({
          clientName = "kiro-oauth-client",
          clientType = "public",
          scopes = { "codewhisperer:completions", "codewhisperer:analysis", "codewhisperer:conversations" },
          grantTypes = { "urn:ietf:params:oauth:grant-type:device_code", "refresh_token" },
          issuerUrl = ISSUER_URL,
        }),
      })
      if reg_err then return start_page("Client registration failed.", region, start_url, method) end
      if reg_resp.status ~= 200 then return start_page("Client registration failed.", region, start_url, method) end
      local reg = json.decode(reg_resp.body)
      if not reg.clientId or not reg.clientSecret then
        return start_page("Client registration response was incomplete.", region, start_url, method)
      end

      local dev_resp, dev_err = client:request({
        method = "POST",
        url = oidc_url(region, "device_authorization"),
        headers = { ["Content-Type"] = "application/json" },
        body = json.encode({ clientId = reg.clientId, clientSecret = reg.clientSecret, startUrl = start_url }),
      })
      if dev_err then return start_page("Device authorization failed.", region, start_url, method) end
      if dev_resp.status ~= 200 then return start_page("Device authorization failed.", region, start_url, method) end
      local dev = json.decode(dev_resp.body)
      if not dev.deviceCode or not dev.userCode then
        return start_page("Device authorization response was incomplete.", region, start_url, method)
      end

      local state = {
        method = method,
        region = region,
        start_url = start_url,
        client_id = reg.clientId,
        client_secret = reg.clientSecret,
        device_code = dev.deviceCode,
        user_code = dev.userCode,
        verification_uri = dev.verificationUri or "",
        verification_uri_complete = dev.verificationUriComplete or "",
        expires_at = os.time() + (dev.expiresIn or 600),
      }
      llm_router.storage.set(scope, "device", state)
      return device_page("", state)
    end

    if action == "poll_device" or action == "submit" then
      local state = llm_router.storage.get(scope, "device")
      if state == nil then
        return start_page("Device login session expired. Start again.", DEFAULT_REGION, BUILDER_START_URL, "builder-id")
      end
      if state.expires_at and os.time() > state.expires_at then
        llm_router.storage.delete(scope, "device")
        return start_page("Device login expired. Start again.", state.region, state.start_url, state.method)
      end
      local client = llm_router.http_client({})
      local resp, err = client:request({
        method = "POST",
        url = oidc_url(state.region, "token"),
        headers = { ["Content-Type"] = "application/json", ["Accept"] = "application/json" },
        body = json.encode({
          clientId = state.client_id,
          clientSecret = state.client_secret,
          deviceCode = state.device_code,
          grantType = "urn:ietf:params:oauth:grant-type:device_code",
        }),
      })
      if err then return device_page("Token request failed.", state) end
      local tok = json.decode(resp.body)
      if resp.status ~= 200 or (tok.error and tok.error ~= "") then
        if tok.error == "authorization_pending" or tok.error == "slow_down" then
          local msg = tok.error_description or tok.error
          if msg == "" then msg = "Authorization is still pending. Finish the browser step, then check again." end
          return device_page(msg, state)
        end
        return device_page(tok.error_description or "Device login failed.", state)
      end
      llm_router.storage.delete(scope, "device")
      local creds = { access_token = tok.accessToken, auth_method = state.method }
      if tok.refreshToken and tok.refreshToken ~= "" then creds.refresh_token = tok.refreshToken end
      if state.region and state.region ~= "" then creds.region = state.region end
      if state.client_id then creds.client_id = state.client_id end
      if state.client_secret then creds.client_secret = state.client_secret end
      if tok.expiresIn and tok.expiresIn > 0 then
        creds.expires_at = os.date("!%Y-%m-%dT%H:%M:%SZ", os.time() + tok.expiresIn)
      end
      return { credentials = creds }
    end

    return start_page("Choose a device login method to continue.", DEFAULT_REGION, BUILDER_START_URL, "builder-id")
  end,

  complete = function(ctx, request)
    local effort = resolve_effort(request)
    local model, thinking, alias_err = resolve_model(request.model_name, effort)
    if alias_err then return nil, alias_err end
    local client = llm_router.http_client({})
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local resp, err = client:request({
          method = "POST",
          url = generate_url(cred, ctx.provider_config),
          headers = generate_headers(cred),
          body = json.encode(build_payload(request, model, thinking, effort)),
          proxy_url = px.url,
        })
        if err == nil and resp.status == 200 then
          return assemble_completion(resp.body, thinking, request.model)
        elseif err ~= nil then
          last_err = err
        else
          local action, terr = map_upstream(ctx, resp, cred.id)
          if action == "done" then return nil, terr end
          last_err = terr
          if action == "cred" then break end
        end
      end
    end
    return nil, last_err or { message = "all kiro credentials exhausted", code = "server_error" }
  end,

  complete_stream = function(ctx, request, emit)
    local model = request.model
    local effort = resolve_effort(request)
    local short_model, thinking, alias_err = resolve_model(request.model_name, effort)
    if alias_err then return nil, alias_err end
    local client = llm_router.http_client({})
    local last_err = nil
    for _, cred in ipairs(llm_router.credentials.list()) do
      for _, px in ipairs(pick_proxies(ctx, 3)) do
        local response_id = "kiro-" .. tostring(os.time())
        local created = os.time()
        local buffer = ""
        local builders = {}
        local saw_tool = false
        local first = true
        local done = false
        local st = { fatal = false, next_cred = false }
        local stream_usage = nil
        local think_state = { thinkingMode = false, pending = "" }
        local function emit_delta(delta)
          if first then delta.role = "assistant" end
          first = false
          emit({
            id = response_id,
            object = "chat.completion.chunk",
            created = created,
            model = model,
            choices = { { index = 0, delta = delta } },
          })
          done = true
        end
        local payload = build_payload(request, short_model, thinking, effort)
        local generate = generate_url(cred, ctx.provider_config)
        local _, stream_err = client_stream_raw(
          generate_headers(cred),
          payload,
          generate,
          px.url,
          cred.id,
          st,
          function(bytes)
            buffer = buffer .. bytes
            while true do
              if #buffer < 16 then break end
              local total = u32be(buffer, 1)
              if total < 16 or total > #buffer then break end
              local frame = parse_frame(buffer:sub(1, total))
              buffer = buffer:sub(total + 1)
              if frame then
                local etype = frame.headers[":event-type"]
                local p = frame.payload
                if etype == "assistantResponseEvent" or etype == "codeEvent" then
                  if type(p.content) == "string" and p.content ~= "" then
                    if thinking then
                      local content_parts, reasoning_parts = {}, {}
                      split_thinking(think_state, p.content, content_parts, reasoning_parts)
                      local text = table.concat(content_parts, "")
                      local reasoning = table.concat(reasoning_parts, "")
                      if text ~= "" then emit_delta({ content = text }) end
                      if reasoning ~= "" then emit_delta({ reasoning_content = reasoning }) end
                    else
                      local delta = { content = p.content }
                      if first then delta.role = "assistant" end
                      first = false
                      emit({
                        id = response_id,
                        object = "chat.completion.chunk",
                        created = created,
                        model = model,
                        choices = { { index = 0, delta = delta } },
                      })
                      done = true
                    end
                  end
                elseif etype == "reasoningContentEvent" then
                  local text = reasoning_text(p)
                  if text ~= "" then emit_delta({ reasoning_content = text }) end
                elseif etype == "toolUseEvent" then
                  local id = p.toolUseId or ""
                  if id == "" then id = "call_" .. tostring(os.time()) end
                  local b = builders[id]
                  if b == nil then
                    b = { name = "", args = "" }
                    builders[id] = b
                  end
                  if type(p.name) == "string" and p.name ~= "" then b.name = p.name end
                  local is_string = false
                  if p.input ~= nil then
                    if type(p.input) == "string" then
                      b.args = b.args .. p.input
                      is_string = true
                    else
                      b.args = json.encode(p.input)
                    end
                  end
                  local should = false
                  if p.stop == true then
                    should = true
                  elseif not is_string and p.input ~= nil and b.name ~= "" then
                    should = true
                  end
                  if should and b.name ~= "" then
                    local args = b.args:match("^%s*(.-)%s*$")
                    if args == "" then args = "{}" end
                    saw_tool = true
                    local delta = {
                      tool_calls = {
                        {
                          index = 0,
                          id = id,
                          type = "function",
                          ["function"] = { name = b.name, arguments = args },
                        },
                      },
                    }
                    if first then delta.role = "assistant" end
                    first = false
                    emit({
                      id = response_id,
                      object = "chat.completion.chunk",
                      created = created,
                      model = model,
                      choices = { { index = 0, delta = delta } },
                    })
                    done = true
                    builders[id] = nil
                  end
                elseif etype == "metricsEvent" then
                  local m = p.metricsEvent or p
                  local itok = tonumber(m.inputTokens) or 0
                  local otok = tonumber(m.outputTokens) or 0
                  if itok > 0 or otok > 0 then
                    stream_usage = { prompt_tokens = itok, completion_tokens = otok, total_tokens = itok + otok }
                  end
                end
              end
            end
          end
        )
        if stream_err then
          last_err = stream_err
          if st.fatal then return nil, last_err end
          if st.next_cred then break end
          -- A stream that emitted already belongs to that attempt: surface
          -- instead of failing over mid-stream.
          if done then return nil, last_err end
        else
          if thinking and think_state.pending ~= "" then
            if think_state.thinkingMode then
              emit_delta({ reasoning_content = think_state.pending })
            else
              emit_delta({ content = think_state.pending })
            end
            think_state.pending = ""
          end
          local finish = "stop"
          if saw_tool then finish = "tool_calls" end
          local last = {
            id = response_id,
            object = "chat.completion.chunk",
            created = created,
            model = model,
            choices = { { index = 0, delta = {}, finish_reason = finish } },
          }
          if stream_usage then last.usage = stream_usage end
          emit(last)
          return
        end
      end
    end
    return nil, last_err or { message = "all kiro credentials exhausted", code = "server_error" }
  end,
})

-- Raw event-stream POST with chunked delivery. Declared after register so
-- the closure above resolves it at call time, not at load time.
function client_stream_raw(headers, payload, url, proxy_url, cred_id, st, on_bytes)
  local client = llm_router.http_client({})
  return client:stream({
    method = "POST",
    url = url,
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
    on_chunk = function(bytes) on_bytes(bytes) end,
  })
end
