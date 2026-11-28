--- @plugin Claude
--- @author TheSlopMachine
--- @version 0.1.2
--- @plugin_api 1.8
--- @description Claude models with an Anthropic API key or with your own Claude subscription. The subscription runs the official Claude Code in a container.
--- @license MIT
--- @allow_host api.anthropic.com
--- @allow_host downloads.claude.ai
--- @allow_host claude.ai
--- @allow_host claude.com
--- @allow_host platform.claude.com
--- @allow_host console.anthropic.com
--- @require_api docker
--- @docker.image ghcr.io/theslopmachine/llm-router-asterisk-bridge:v0.1.0@sha256:1240a7085bc06f7a45f857c402655ad2f0bfb9cdf4c7373fe18d463d2978acb5
--- @docker.memory_limit_mb 3072
--- @docker.cpu_limit 2
--- @docker.pids_limit 1024
--- @docker.container_limit 1

-- Two kinds of credentials serve the same models:
--   API key: the plugin calls the Messages API directly.
--   Subscription: the asterisk bridge (github.com/TheSlopMachine/llm-router-asterisk-bridge)
--   runs the official Claude Code signed in with the account. The bridge
--   keeps one live Claude Code per conversation, so a continued
--   conversation sends only its new messages. Client tools reach Claude
--   Code through a relay MCP server; Claude Code has no tools of its own.
-- Credential data: { api_key } or { account, email }. The sign-in of a
-- subscription account stays inside the container: the router holds only
-- the account slot name.

local docker = llm_router.docker

local API_URL = "https://api.anthropic.com/v1"
local API_VERSION = "2023-06-01"
local IMAGE = "ghcr.io/theslopmachine/llm-router-asterisk-bridge:v0.1.0@sha256:1240a7085bc06f7a45f857c402655ad2f0bfb9cdf4c7373fe18d463d2978acb5"
-- Claude Code release the bridge installs. Every flag the bridge passes
-- exists in this release.
local CLAUDE_CODE_VERSION = "2.1.296"
local KEY = "bridge"
local PORT = 8080
local STATE_SCOPE = "container"
local FLOW_SCOPE = "sign_in"
local STREAM_TIMEOUT_MS = 900000
local DEFAULT_PARK_SECONDS = 60
local DEAD_PARK_SECONDS = 3600
local MAX_PARK_SECONDS = 90 * 24 * 3600
-- Reply budget when the client sets none.
local DEFAULT_MAX_TOKENS = 32000

-- ── Shared helpers ──

-- The plugin Lua state keeps a fixed-size value stack, so one table.concat
-- over thousands of fragments overflows it. Grouped joining keeps the
-- operand count flat.
local JOIN_GROUP_SIZE = 256

local function join_parts(parts)
  local count = #parts
  if count <= JOIN_GROUP_SIZE then return table.concat(parts, "") end
  local groups = {}
  local index = 1
  while index <= count do
    local stop = math.min(index + JOIN_GROUP_SIZE - 1, count)
    local pieces = {}
    for position = index, stop do pieces[#pieces + 1] = parts[position] end
    groups[#groups + 1] = table.concat(pieces, "")
    index = stop + 1
  end
  return join_parts(groups)
end

local function trim(value)
  if type(value) ~= "string" then return "" end
  return (string.gsub(value, "^%s*(.-)%s*$", "%1"))
end

local function is_empty_table(value)
  return type(value) == "table" and next(value) == nil
end

local function decode(body)
  local ok, decoded = pcall(json.decode, body or "")
  if ok and type(decoded) == "table" then return decoded end
  return nil
end

local function kind_of(credential_entry)
  local data = credential_entry.data or {}
  if type(data.account) == "string" and data.account ~= "" then return "subscription" end
  if type(data.api_key) == "string" and data.api_key ~= "" then return "api_key" end
  return nil
end

local function park(credential_entry, seconds, reason)
  seconds = math.min(math.max(math.floor(seconds or DEFAULT_PARK_SECONDS), 1), MAX_PARK_SECONDS)
  pcall(llm_router.credentials.park, credential_entry.id, seconds, reason)
  llm_router.log("warn", "credential parked", { cred_id = credential_entry.id, wait = seconds, reason = reason })
end

-- A dead credential is disabled when the provider allows it, parked
-- otherwise.
local function retire(ctx, credential_entry, reason)
  local config = ctx.provider_config
  if type(config) == "table" and config.disable_failed_credentials == true
    and pcall(llm_router.credentials.disable, credential_entry.id, reason) then
    llm_router.log("warn", "credential disabled", { cred_id = credential_entry.id, reason = reason })
    return
  end
  park(credential_entry, DEAD_PARK_SECONDS, reason)
end

-- Credentials in the order the provider configures: subscription accounts
-- first by default, they cost no API credit.
local function ordered_credentials(ctx)
  local subscriptions, api_keys = {}, {}
  for _, credential_entry in ipairs(llm_router.credentials.list()) do
    local kind = kind_of(credential_entry)
    if kind == "subscription" then
      table.insert(subscriptions, credential_entry)
    elseif kind == "api_key" then
      table.insert(api_keys, credential_entry)
    end
  end
  local config = ctx.provider_config
  local api_first = type(config) == "table" and config.credential_order == "api_key_first"
  return subscriptions, api_keys, api_first
end

-- JSON Schema keywords that hold arrays. An empty Lua table encodes as an
-- object, so these turn back into arrays before a schema leaves the plugin.
local SCHEMA_ARRAYS = { required = true, enum = true, anyOf = true, oneOf = true, allOf = true, prefixItems = true, examples = true, type = true }

local function normalize_schema(schema)
  if type(schema) ~= "table" then return schema end
  local result = {}
  for key, value in pairs(schema) do
    if SCHEMA_ARRAYS[key] and is_empty_table(value) then
      result[key] = json.array({})
    elseif type(value) == "table" then
      result[key] = normalize_schema(value)
    else
      result[key] = value
    end
  end
  return result
end

local function object_schema(parameters)
  local schema = normalize_schema(parameters or {})
  if schema.type == nil then schema.type = "object" end
  if schema.type == "object" and schema.properties == nil then schema.properties = {} end
  return schema
end

local function tool_spec(tool)
  local spec = tool["function"] or tool
  return {
    name = spec.name,
    description = spec.description,
    parameters = object_schema(spec.parameters or spec.input_schema),
  }
end

-- ── Container ──

local function state_token()
  local stored = llm_router.storage.get(STATE_SCOPE, "token")
  if type(stored) ~= "string" or #stored < 32 then
    stored = llm_router.random_hex(32)
    llm_router.storage.set(STATE_SCOPE, "token", stored)
  end
  return stored
end

local function container_spec()
  return {
    image = IMAGE,
    ports = { PORT },
    environment = {
      ASTERISK_TOKEN = state_token(),
      ASTERISK_CLAUDE_VERSION = CLAUDE_CODE_VERSION,
    },
    volumes = { ["bridge-data"] = "/data" },
    restart = "on_failure",
  }
end

local function stopped(fault)
  return fault ~= nil and fault.docker and (fault.reason == "not_running" or fault.reason == "not_found")
end

-- Creates, starts and waits for the container. Every step is idempotent.
local function ensure_running(timeout_ms)
  local status, fault = docker.ensure(KEY, container_spec())
  if fault ~= nil then return fault end
  if status.state == "preparing" then
    return {
      message = "the bridge image is downloading, try again in a minute",
      code = "server_error", status = 503, docker = true, reason = "preparing",
    }
  end
  if status.state ~= "running" then
    local _, start_fault = docker.start(KEY)
    if start_fault ~= nil then return start_fault end
  end
  local _, wait_fault = docker.wait(KEY, { port = PORT, timeout_ms = timeout_ms or 60000 })
  return wait_fault
end

local function bridge_headers()
  return { authorization = "Bearer " .. state_token(), ["content-type"] = "application/json" }
end

-- Runs one call against the container and starts a stopped container once.
local function with_container(run)
  local result, fault = run()
  if not stopped(fault) then return result, fault end
  local start_fault = ensure_running()
  if start_fault ~= nil then return nil, start_fault end
  return run()
end

local function bridge_call(method, path, body, timeout_ms)
  local response, fault = with_container(function()
    local client = docker.open_http(KEY, PORT, { timeout_ms = timeout_ms or 60000 })
    return client:request({ method = method, url = path, headers = bridge_headers(), body = body })
  end)
  if fault ~= nil then return nil, nil, fault end
  return decode(response.body) or {}, response.status, nil
end

-- The bridge answers {"error": {code, message, account, reset_at}}.
local function bridge_failure(decoded, status)
  if type(decoded) == "table" and type(decoded.error) == "table" then return decoded.error end
  return { code = "upstream_error", message = "the bridge answered status " .. tostring(status) }
end

-- Bridge codes that concern one account, with the action they take.
local ACCOUNT_ACTIONS = {
  rate_limit = "park",
  billing = "park",
  not_signed_in = "retire",
}

local function router_error(failure)
  local code = failure.code
  local message = "subscription: " .. tostring(failure.message or code)
  if code == "rate_limit" then
    local result = { message = message, code = "rate_limit", status = 429 }
    if type(failure.reset_at) == "number" and failure.reset_at > os.time() then
      result.headers = { ["Retry-After"] = tostring(failure.reset_at - os.time()) }
    end
    return result
  end
  if code == "not_signed_in" then return { message = message, code = "auth_error", status = 401 } end
  if code == "billing" then return { message = message, code = "payment_required", status = 402 } end
  if code == "invalid_request" then return { message = message, code = "invalid_request_error", status = 400 } end
  if code == "busy" or code == "overloaded" then return { message = message, code = "overloaded", status = 503 } end
  if code == "not_ready" then return { message = message, code = "server_error", status = 503 } end
  return { message = message, code = "upstream_error", status = 502 }
end

-- ── Messages API: request ──

local function image_source(url)
  if type(url) ~= "string" then return nil end
  if string.match(url, "^https?://") then return { type = "url", url = url } end
  local media_type, data = string.match(url, "^data:([^;,]+);base64,(.*)$")
  if media_type == nil then return nil end
  return { type = "base64", media_type = media_type, data = data }
end

local function document_block(file)
  local media_type, data = string.match(file.file_data or "", "^data:([^;,]+)[^,]*;base64,(.*)$")
  if media_type == nil then return nil end
  local block
  if media_type == "application/pdf" then
    block = { type = "document", source = { type = "base64", media_type = media_type, data = data } }
  elseif string.sub(media_type, 1, 5) == "text/" then
    block = { type = "document", source = { type = "text", media_type = "text/plain", data = llm_router.base64_decode(data) } }
  else
    return nil
  end
  if type(file.filename) == "string" and file.filename ~= "" then block.title = file.filename end
  return block
end

local function invalid(message)
  return { message = message, code = "invalid_request_error", status = 400 }
end

-- OpenAI content (string or part array) to Messages API blocks.
local function content_blocks(content)
  if content == nil then return {}, nil end
  if type(content) == "string" then
    if content == "" then return {}, nil end
    return { { type = "text", text = content } }, nil
  end
  local blocks = {}
  for _, part in ipairs(content) do
    if part.type == "text" then
      if type(part.text) == "string" and part.text ~= "" then table.insert(blocks, { type = "text", text = part.text }) end
    elseif part.type == "image_url" then
      local source = image_source(part.image_url and part.image_url.url)
      if source == nil then return nil, invalid("image_url must be an http(s) or a base64 data URL") end
      table.insert(blocks, { type = "image", source = source })
    elseif part.type == "file" then
      local block = document_block(part.file or {})
      if block == nil then return nil, invalid("file parts must be PDF or text data URLs") end
      table.insert(blocks, block)
    else
      return nil, invalid("Claude does not take " .. tostring(part.type) .. " parts")
    end
  end
  return blocks, nil
end

local function text_of(content)
  if type(content) == "string" then return content end
  local parts = {}
  for _, part in ipairs(content or {}) do
    if part.type == "text" and type(part.text) == "string" then table.insert(parts, part.text) end
  end
  return table.concat(parts, "\n")
end

-- Appends blocks as a message, merging with the previous message of the
-- same role: the API wants roles to alternate and every tool result in the
-- user message right after the calls.
local function push(messages, role, blocks)
  if #blocks == 0 then return end
  local last = messages[#messages]
  if last ~= nil and last.role == role then
    for _, block in ipairs(blocks) do table.insert(last.content, block) end
    return
  end
  table.insert(messages, { role = role, content = blocks })
end

local function api_messages(request)
  local system_parts = {}
  local messages = {}
  local leading = true
  for _, message in ipairs(request.messages or {}) do
    local role = message.role
    if role == "system" or role == "developer" then
      if leading then
        table.insert(system_parts, text_of(message.content))
      else
        push(messages, "user", { { type = "text", text = "<system>\n" .. text_of(message.content) .. "\n</system>" } })
      end
    else
      leading = false
      if role == "assistant" then
        local blocks = {}
        local text = text_of(message.content)
        if trim(text) ~= "" then table.insert(blocks, { type = "text", text = text }) end
        for _, call in ipairs(message.tool_calls or {}) do
          local fn = call["function"] or {}
          local input = decode(fn.arguments) or {}
          table.insert(blocks, { type = "tool_use", id = call.id, name = fn.name, input = input })
        end
        push(messages, "assistant", blocks)
      elseif role == "tool" then
        local result = { type = "tool_result", tool_use_id = message.tool_call_id }
        if type(message.content) == "string" then
          result.content = message.content
        else
          local blocks, problem = content_blocks(message.content)
          if problem ~= nil then return nil, nil, problem end
          result.content = blocks
        end
        push(messages, "user", { result })
      else
        local blocks, problem = content_blocks(message.content)
        if problem ~= nil then return nil, nil, problem end
        push(messages, "user", blocks)
      end
    end
  end
  return table.concat(system_parts, "\n\n"), messages, nil
end

-- Forced tool choice returns 400 on these models: they take auto only.
local AUTO_ONLY_TOOL_CHOICE = { "^claude%-opus%-5%-5", "^claude%-sonnet%-5%-5", "^claude%-fable%-5%-[12]", "^claude%-mythos%-5%-[12]" }

local function forced_tool_choice_allowed(model)
  for _, pattern in ipairs(AUTO_ONLY_TOOL_CHOICE) do
    if string.match(model, pattern) then return false end
  end
  return true
end

-- Adaptive thinking and effort exist from the 4.6 models on.
local function supports_adaptive(model)
  if string.match(model, "^claude%-3") or string.match(model, "%-4%-5") or string.match(model, "%-4%-1")
    or string.match(model, "%-4$") or string.match(model, "%-4%-0") then
    return false
  end
  return true
end

local EFFORTS = { low = "low", medium = "medium", high = "high", xhigh = "xhigh", max = "max", minimal = "low", none = "low" }

local function api_payload(request, stream)
  local system, messages, problem = api_messages(request)
  if problem ~= nil then return nil, problem end
  if #messages == 0 then return nil, invalid("the conversation has no user message") end
  local model = request.model_name
  local payload = {
    model = model,
    messages = messages,
    max_tokens = request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_TOKENS,
    stream = stream or nil,
    cache_control = { type = "ephemeral" },
  }
  if system ~= "" then payload.system = system end
  if type(request.stop) == "string" then
    payload.stop_sequences = { request.stop }
  elseif type(request.stop) == "table" and #request.stop > 0 then
    payload.stop_sequences = request.stop
  end
  if type(request.tools) == "table" and #request.tools > 0 then
    local tools = {}
    for _, tool in ipairs(request.tools) do
      local spec = tool_spec(tool)
      table.insert(tools, { name = spec.name, description = spec.description, input_schema = spec.parameters })
    end
    payload.tools = tools
    local choice = request.tool_choice
    local forced = forced_tool_choice_allowed(model)
    if choice == "none" then
      payload.tool_choice = { type = "none" }
    elseif choice == "required" and forced then
      payload.tool_choice = { type = "any" }
    elseif type(choice) == "table" and choice["function"] ~= nil and forced then
      payload.tool_choice = { type = "tool", name = choice["function"].name }
    elseif request.parallel_tool_calls == false then
      payload.tool_choice = { type = "auto" }
    end
    if request.parallel_tool_calls == false and payload.tool_choice ~= nil and payload.tool_choice.type ~= "none" then
      payload.tool_choice.disable_parallel_tool_use = true
    end
  end
  local output_config = {}
  local effort = EFFORTS[request.reasoning_effort or ""]
  if effort ~= nil and supports_adaptive(model) then
    output_config.effort = effort
    if request.reasoning_effort ~= "none" and request.reasoning_effort ~= "minimal" then
      payload.thinking = { type = "adaptive", display = "summarized" }
    end
  end
  local format = request.response_format
  if type(format) == "table" and format.type == "json_schema" and type(format.json_schema) == "table" then
    output_config.format = { type = "json_schema", schema = normalize_schema(format.json_schema.schema or format.json_schema) }
  end
  if next(output_config) ~= nil then payload.output_config = output_config end
  -- The API rejects thinking together with a forced tool call.
  if payload.tool_choice ~= nil and (payload.tool_choice.type == "any" or payload.tool_choice.type == "tool") then
    payload.thinking = nil
  end
  return payload, nil
end

-- ── Messages API: response ──

local FINISH_REASONS = { end_turn = "stop", stop_sequence = "stop", max_tokens = "length", tool_use = "tool_calls", refusal = "content_filter", pause_turn = "stop" }

local function openai_usage(usage)
  usage = usage or {}
  local input = (usage.input_tokens or 0) + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
  local output = usage.output_tokens or 0
  return {
    prompt_tokens = input,
    completion_tokens = output,
    total_tokens = input + output,
    prompt_tokens_details = { cached_tokens = usage.cache_read_input_tokens or 0 },
  }
end

local function openai_response(body, request)
  local text, reasoning, calls = {}, {}, {}
  for _, block in ipairs(body.content or {}) do
    if block.type == "text" then
      table.insert(text, block.text or "")
    elseif block.type == "thinking" then
      table.insert(reasoning, block.thinking or "")
    elseif block.type == "tool_use" then
      table.insert(calls, { id = block.id, type = "function", ["function"] = { name = block.name, arguments = json.encode(block.input or {}) } })
    end
  end
  local message = { role = "assistant", content = join_parts(text) }
  local reasoning_text = join_parts(reasoning)
  if reasoning_text ~= "" then message.reasoning_content = reasoning_text end
  if #calls > 0 then message.tool_calls = calls end
  return {
    id = body.id or ("chatcmpl-" .. llm_router.random_hex(12)),
    object = "chat.completion",
    created = os.time(),
    model = request.model,
    choices = { { index = 0, message = message, finish_reason = FINISH_REASONS[body.stop_reason or ""] or "stop" } },
    usage = openai_usage(body.usage),
  }
end

-- What a non-200 API answer means for the credential:
--   "next": try the next credential after the action already taken.
--   "fail": return the error.
local function api_outcome(ctx, credential_entry, status, headers, body)
  local decoded = decode(body) or {}
  local detail = type(decoded.error) == "table" and decoded.error or {}
  local message = "anthropic status " .. tostring(status) .. ": " .. tostring(detail.message or string.sub(body or "", 1, 500))
  if status == 401 or status == 403 then
    retire(ctx, credential_entry, "rejected key")
    return "next", { message = message, code = "auth_error", status = status }
  end
  if status == 429 then
    park(credential_entry, tonumber((headers or {})["retry-after"]) or DEFAULT_PARK_SECONDS, "rate_limit")
    return "next", { message = message, code = "rate_limit", status = 429 }
  end
  if status == 402 then
    park(credential_entry, DEAD_PARK_SECONDS, "payment_required")
    return "next", { message = message, code = "payment_required", status = 402 }
  end
  if status == 404 and detail.type == "not_found_error" then
    return "fail", { message = message, code = "not_found", status = 404 }
  end
  if status == 400 or status == 413 or status == 404 then
    return "fail", { message = message, code = "invalid_request_error", status = status }
  end
  if status == 529 then
    return "fail", { message = message, code = "overloaded", status = 529 }
  end
  return "next", { message = message, code = "upstream_error", status = 502 }
end

local function api_headers(credential_entry)
  return {
    ["content-type"] = "application/json",
    ["x-api-key"] = credential_entry.data.api_key,
    ["anthropic-version"] = API_VERSION,
  }
end

local function api_complete(ctx, request, credentials)
  local payload, problem = api_payload(request, false)
  if problem ~= nil then return nil, problem, false end
  local body = json.encode(payload)
  local client = llm_router.http_client({ timeout_ms = STREAM_TIMEOUT_MS })
  local last_error = nil
  for _, credential_entry in ipairs(credentials) do
    local response, fault = client:request({ method = "POST", url = API_URL .. "/messages", headers = api_headers(credential_entry), body = body })
    if fault ~= nil then
      last_error = fault
    elseif response.status == 200 then
      return openai_response(json.decode(response.body), request), nil, false
    else
      local outcome, failure = api_outcome(ctx, credential_entry, response.status, response.headers, response.body)
      if outcome == "fail" then return nil, failure, false end
      last_error = failure
    end
  end
  return nil, last_error, true
end

-- Streams one Messages API call as OpenAI chunks.
local function api_stream(ctx, request, credentials, emit)
  local payload, problem = api_payload(request, true)
  if problem ~= nil then return problem, false end
  local body = json.encode(payload)
  local client = llm_router.http_client({ timeout_ms = STREAM_TIMEOUT_MS })
  local last_error = nil
  for _, credential_entry in ipairs(credentials) do
    local id = "chatcmpl-" .. llm_router.random_hex(12)
    local created = os.time()
    local started = false
    local rejected = nil
    local stream_error = nil
    -- block_calls maps a stream block index to its tool call index.
    local block_calls = {}
    local call_arguments = {}
    local call_count = 0
    local usage = {}
    local stop_reason = nil

    local function chunk(delta, finish)
      if not started then
        delta.role = "assistant"
        started = true
      end
      return { id = id, object = "chat.completion.chunk", created = created, model = request.model,
        choices = { { index = 0, delta = delta, finish_reason = finish } } }
    end

    local _, fault = client:stream({
      method = "POST", url = API_URL .. "/messages", headers = api_headers(credential_entry), body = body,
      on_response = function(head)
        if head.status ~= 200 then
          rejected = head
          return { message = "anthropic status " .. tostring(head.status), code = "upstream_error", status = 502 }
        end
      end,
      on_line = function(line)
        local data = string.match(line, "^data: (.*)$")
        if data == nil then return end
        local event = decode(data)
        if event == nil then return end
        if event.type == "message_start" then
          local start = (event.message or {}).usage or {}
          usage.input_tokens = start.input_tokens or 0
          usage.cache_read_input_tokens = start.cache_read_input_tokens or 0
          usage.cache_creation_input_tokens = start.cache_creation_input_tokens or 0
        elseif event.type == "content_block_start" then
          local block = event.content_block or {}
          if block.type == "tool_use" then
            local index = call_count
            block_calls[event.index] = index
            call_count = call_count + 1
            emit(chunk({ tool_calls = { { index = index, id = block.id, type = "function", ["function"] = { name = block.name, arguments = "" } } } }))
          end
        elseif event.type == "content_block_delta" then
          local delta = event.delta or {}
          if delta.type == "text_delta" and delta.text ~= "" then
            emit(chunk({ content = delta.text }))
          elseif delta.type == "thinking_delta" and (delta.thinking or "") ~= "" then
            emit(chunk({ reasoning_content = delta.thinking }))
          elseif delta.type == "input_json_delta" and (delta.partial_json or "") ~= "" and block_calls[event.index] ~= nil then
            call_arguments[block_calls[event.index]] = true
            emit(chunk({ tool_calls = { { index = block_calls[event.index], ["function"] = { arguments = delta.partial_json } } } }))
          end
        elseif event.type == "message_delta" then
          if event.delta ~= nil and event.delta.stop_reason ~= nil then stop_reason = event.delta.stop_reason end
          if event.usage ~= nil then usage.output_tokens = event.usage.output_tokens end
        elseif event.type == "error" then
          stream_error = event.error or {}
        end
      end,
    })
    if rejected ~= nil then
      local outcome, failure = api_outcome(ctx, credential_entry, rejected.status, rejected.headers, rejected.body)
      if outcome == "fail" then return failure, false end
      last_error = failure
    elseif fault ~= nil and not started then
      last_error = fault
    elseif fault ~= nil then
      return fault, false
    elseif stream_error ~= nil then
      local failure = { message = "anthropic stream error: " .. tostring(stream_error.message or stream_error.type), code = "upstream_error", status = 502 }
      if stream_error.type == "overloaded_error" then failure.code = "overloaded" end
      if started then return failure, false end
      last_error = failure
    else
      -- A call without input streams no arguments: clients expect "{}".
      for index = 0, call_count - 1 do
        if not call_arguments[index] then
          emit(chunk({ tool_calls = { { index = index, ["function"] = { arguments = "{}" } } } }))
        end
      end
      emit(chunk({}, FINISH_REASONS[stop_reason or ""] or "stop"))
      emit({ id = id, object = "chat.completion.chunk", created = created, model = request.model, choices = json.array({}), usage = openai_usage(usage) })
      return nil, false
    end
  end
  return last_error, true
end

-- ── Subscription path ──

local function bridge_body(request, stream, slots)
  local messages = {}
  for _, message in ipairs(request.messages or {}) do
    local copy = {}
    for key, value in pairs(message) do copy[key] = value end
    if is_empty_table(copy.content) then copy.content = "" end
    if is_empty_table(copy.tool_calls) then copy.tool_calls = nil end
    table.insert(messages, copy)
  end
  local tools = nil
  if type(request.tools) == "table" and #request.tools > 0 then
    tools = {}
    for _, tool in ipairs(request.tools) do
      local spec = tool_spec(tool)
      table.insert(tools, { type = "function", ["function"] = spec })
    end
  end
  return json.encode({
    model = request.model_name,
    messages = messages,
    tools = tools,
    tool_choice = request.tool_choice,
    response_format = request.response_format,
    reasoning_effort = request.reasoning_effort,
    stream = stream or nil,
    accounts = json.array(slots),
  })
end

-- Applies an account failure to the credentials of that slot. Returns true
-- when the failure was about one account, so the next account may answer.
local function account_failure(ctx, remaining, failure)
  local action = ACCOUNT_ACTIONS[failure.code or ""]
  if action == nil or type(failure.account) ~= "string" then return false end
  for index, credential_entry in ipairs(remaining) do
    if credential_entry.data.account == failure.account then
      if action == "retire" then
        retire(ctx, credential_entry, "sign in again")
      else
        local wait = DEFAULT_PARK_SECONDS
        if failure.code == "billing" then wait = DEAD_PARK_SECONDS end
        if type(failure.reset_at) == "number" and failure.reset_at > os.time() then wait = failure.reset_at - os.time() end
        park(credential_entry, wait, failure.code)
      end
      table.remove(remaining, index)
      return true
    end
  end
  return false
end

local function slots_of(credentials)
  local slots = {}
  for _, credential_entry in ipairs(credentials) do table.insert(slots, credential_entry.data.account) end
  return slots
end

-- Codes after which an API key may still answer.
local FALL_THROUGH = { busy = true, not_ready = true, overloaded = true, upstream_error = true }

local function subscription_complete(ctx, request, credentials)
  local remaining = {}
  for _, credential_entry in ipairs(credentials) do table.insert(remaining, credential_entry) end
  local last_error = nil
  while #remaining > 0 do
    local decoded, status, fault = bridge_call("POST", "/v1/chat/completions", bridge_body(request, false, slots_of(remaining)), STREAM_TIMEOUT_MS)
    if fault ~= nil then return nil, fault, true end
    if status == 200 then return decoded, nil, false end
    local failure = bridge_failure(decoded, status)
    last_error = router_error(failure)
    if not account_failure(ctx, remaining, failure) then
      return nil, last_error, FALL_THROUGH[failure.code or ""] == true
    end
  end
  return nil, last_error, true
end

local function subscription_stream(ctx, request, credentials, emit)
  local remaining = {}
  for _, credential_entry in ipairs(credentials) do table.insert(remaining, credential_entry) end
  local last_error = nil
  while #remaining > 0 do
    local rejected = nil
    local stream_failure = nil
    local started = false
    local body = bridge_body(request, true, slots_of(remaining))
    local _, fault = with_container(function()
      local client = docker.open_http(KEY, PORT, { timeout_ms = STREAM_TIMEOUT_MS })
      return client:stream({
        method = "POST", url = "/v1/chat/completions", headers = bridge_headers(), body = body,
        on_response = function(head)
          if head.status ~= 200 then
            rejected = bridge_failure(decode(head.body), head.status)
            return router_error(rejected)
          end
        end,
        on_line = function(line)
          local data = string.match(line, "^data: (.*)$")
          if data == nil or data == "[DONE]" then return end
          local chunk = decode(data)
          if chunk == nil then return end
          if type(chunk.error) == "table" then
            stream_failure = chunk.error
            return
          end
          if (chunk.choices ~= nil and #chunk.choices > 0) or chunk.usage ~= nil then
            started = true
            emit(chunk)
          end
        end,
      })
    end)
    if stream_failure ~= nil then
      if started then return router_error(stream_failure), false end
      rejected = stream_failure
    end
    if rejected ~= nil then
      last_error = router_error(rejected)
      if not account_failure(ctx, remaining, rejected) then
        return last_error, FALL_THROUGH[rejected.code or ""] == true
      end
    elseif fault ~= nil then
      return fault, not started
    else
      return nil, false
    end
  end
  return last_error, true
end

-- ── Request entry ──

-- Runs the two credential groups in the configured order. The second group
-- runs only when the first one allows a fall-through and nothing streamed.
local function serve(ctx, run_subscription, run_api)
  local subscriptions, api_keys, api_first = ordered_credentials(ctx)
  local order = { { "subscription", subscriptions }, { "api_key", api_keys } }
  if api_first then order = { order[2], order[1] } end
  local last_error = nil
  for _, group in ipairs(order) do
    if #group[2] > 0 then
      local result, failure, fall_through
      if group[1] == "subscription" then
        result, failure, fall_through = run_subscription(group[2])
      else
        result, failure, fall_through = run_api(group[2])
      end
      if failure == nil then return result, nil end
      last_error = failure
      if not fall_through then return nil, failure end
      llm_router.log("warn", "credential group failed", { group = group[1], code = failure.code, error = failure.message })
    end
  end
  if last_error ~= nil then return nil, last_error end
  return nil, { message = "no Claude credential is usable", code = "auth_error", status = 401 }
end

-- ── Model cards ──

local SUBSCRIPTION_MODELS = {
  { name = "claude-opus-5-5", display_name = "Claude Opus 5.5" },
  { name = "claude-sonnet-5-5", display_name = "Claude Sonnet 5.5" },
  { name = "claude-haiku-5-5", display_name = "Claude Haiku 5.5" },
  { name = "claude-fable-5-2", display_name = "Claude Fable 5.2" },
}

local function model_card(name, display_name, context_window, max_tokens)
  local card = {
    name = name,
    display_name = display_name or name,
    context_window = context_window,
    max_tokens = max_tokens,
    input_modalities = { "text", "image", "file" },
    output_modalities = { "text" },
    capabilities = { "tools" },
    endpoints = { "chat/completions" },
  }
  if supports_adaptive(name) then
    card.capabilities = { "tools", "reasoning", "structured_outputs" }
    card.reasoning = { supported_efforts = { "max", "xhigh", "high", "medium", "low" }, default_enabled = true }
  end
  return card
end

local function api_models(credential_entry)
  local client = llm_router.http_client({ timeout_ms = 30000 })
  local response, fault = client:request({ method = "GET", url = API_URL .. "/models?limit=1000", headers = api_headers(credential_entry) })
  if fault ~= nil then return nil, fault end
  if response.status ~= 200 then
    return nil, { message = "anthropic models status " .. tostring(response.status), code = "upstream_error", status = 502 }
  end
  local cards = {}
  for _, entry in ipairs((json.decode(response.body) or {}).data or {}) do
    table.insert(cards, model_card(entry.id, entry.display_name, entry.max_input_tokens, entry.max_tokens))
  end
  return cards, nil
end

-- ── Sign-in wizard ──

-- Device login signs in a subscription account. API keys use the plain
-- credential form.

local function error_nodes(error_key, detail)
  local nodes = {}
  if error_key ~= nil then
    table.insert(nodes, { type = "banner", variant = "error", text = error_key })
    if detail ~= nil and detail ~= "" then table.insert(nodes, { type = "code", text = detail }) end
  end
  return nodes
end

-- Shown when no sign-in link could be opened.
local function retry_page(error_key, detail)
  local nodes = error_nodes(error_key, detail)
  table.insert(nodes, {
    type = "section",
    title = "claude.sign_in.title",
    subtitle = "claude.sign_in.subtitle",
    content = {
      { type = "button", text = "claude.action.try_again", form_action = "restart", variant = "primary" },
    },
  })
  return { render = nodes }
end

local function code_page(url, error_key, detail)
  local nodes = error_nodes(error_key, detail)
  table.insert(nodes, {
    type = "section",
    title = "claude.sign_in.title",
    subtitle = "claude.sign_in.subtitle",
    content = {
      { type = "banner", variant = "info", text = "claude.sign_in.own_use" },
      { type = "text", text = "claude.sign_in.step_open" },
      { type = "link", text = "claude.sign_in.open", url = url },
      { type = "text", text = "claude.sign_in.step_code" },
      { type = "input", name = "code", label = "claude.field.code", required = true },
      { type = "button", text = "claude.action.start_over", form_action = "restart", variant = "secondary" },
      { type = "button", text = "claude.action.finish", form_action = "submit_code", variant = "primary" },
    },
  })
  return { render = nodes }
end

local function flow_slot(ctx)
  return llm_router.storage.get(FLOW_SCOPE, ctx.flow_id or "")
end

-- Drops an unfinished sign-in and its empty account.
local function abandon_flow(ctx)
  local slot = flow_slot(ctx)
  if type(slot) ~= "string" then return end
  llm_router.storage.delete(FLOW_SCOPE, ctx.flow_id)
  bridge_call("DELETE", "/accounts/" .. slot, nil, 30000)
end

local function start_subscription(ctx)
  abandon_flow(ctx)
  local fault = ensure_running(90000)
  if fault ~= nil then
    if fault.reason == "preparing" then return retry_page("claude.error.preparing") end
    return retry_page("claude.error.container", fault.message)
  end
  local slot = llm_router.random_hex(8)
  local decoded, status, call_fault = bridge_call("POST", "/accounts/" .. slot .. "/login", nil, 90000)
  if call_fault ~= nil then return retry_page("claude.error.container", call_fault.message) end
  if status ~= 200 then
    local failure = bridge_failure(decoded, status)
    if failure.code == "not_ready" then return retry_page("claude.error.preparing", failure.message) end
    return retry_page("claude.error.sign_in_failed", failure.message)
  end
  llm_router.storage.set(FLOW_SCOPE, ctx.flow_id, slot, { ttl = 1800 })
  llm_router.storage.set(FLOW_SCOPE, ctx.flow_id .. ":url", decoded.url, { ttl = 1800 })
  return code_page(decoded.url)
end

local function submit_code(ctx, values)
  local slot = flow_slot(ctx)
  local url = llm_router.storage.get(FLOW_SCOPE, (ctx.flow_id or "") .. ":url")
  if type(slot) ~= "string" or type(url) ~= "string" then return retry_page("claude.error.expired") end
  local code = trim(values.code)
  if code == "" then return code_page(url, "claude.error.missing_code") end
  local decoded, status, fault = bridge_call("POST", "/accounts/" .. slot .. "/login/code", json.encode({ code = code }), 150000)
  if fault ~= nil then return code_page(url, "claude.error.container", fault.message) end
  if status ~= 200 then
    local failure = bridge_failure(decoded, status)
    if failure.code == "not_found" then return retry_page("claude.error.expired") end
    return code_page(url, "claude.error.sign_in_failed", failure.message)
  end
  if decoded.state ~= "succeeded" then
    -- The bridge ended this sign-in: a new one needs a new link.
    abandon_flow(ctx)
    return retry_page("claude.error.sign_in_failed", decoded.message)
  end
  llm_router.storage.delete(FLOW_SCOPE, ctx.flow_id)
  llm_router.storage.delete(FLOW_SCOPE, ctx.flow_id .. ":url")
  local account, _, _ = bridge_call("GET", "/accounts/" .. slot, nil, 30000)
  local email = ""
  if type(account) == "table" and type(account.status) == "table" and type(account.status.email) == "string" then
    email = account.status.email
  end
  llm_router.log("info", "subscription account signed in", { operation = "auth_step" })
  return { credentials = { account = slot, email = email } }
end

-- ── Health ──

local function subscription_health(credential_entry)
  local decoded, status, fault = bridge_call("GET", "/accounts/" .. credential_entry.data.account, nil, 45000)
  if fault ~= nil then return { status = "unknown", message = fault.message } end
  if status ~= 200 then return { status = "unknown", message = bridge_failure(decoded, status).message } end
  local account = decoded.status or {}
  if account.logged_in == true and account.auth_method == "claude.ai" then return { status = "healthy" } end
  return { status = "unhealthy", message = "the account is not signed in: sign in again" }
end

local function api_health(credential_entry)
  local client = llm_router.http_client({ timeout_ms = 15000 })
  local response, fault = client:request({ method = "GET", url = API_URL .. "/models?limit=1", headers = api_headers(credential_entry) })
  if fault ~= nil then return { status = "unknown", message = "probe failed" } end
  if response.status == 200 then return { status = "healthy" } end
  if response.status == 401 or response.status == 403 then return { status = "unhealthy", message = "rejected key" } end
  return { status = "unknown", message = "status " .. tostring(response.status) }
end

-- One probe generation per credential: healthy wins, quota outranks
-- unknown, a missing model needs every API key to agree.
local function probe_model(ctx, model_ref)
  local request = {
    model = model_ref.model, model_name = model_ref.model_name, max_tokens = 16,
    messages = { { role = "user", content = "Reply with one word: ok" } },
  }
  local subscriptions, api_keys = ordered_credentials(ctx)
  local seen_quota, missing, tried, last = false, 0, 0, "no credential answered"
  for _, credential_entry in ipairs(subscriptions) do
    local decoded, status, fault = bridge_call("POST", "/v1/chat/completions", bridge_body(request, false, { credential_entry.data.account }), 300000)
    if fault == nil and status == 200 then return { status = "healthy", response = "ok" } end
    local failure = fault or bridge_failure(decoded, status)
    if failure.code == "rate_limit" then seen_quota = true end
    last = tostring(failure.message)
  end
  local payload = api_payload(request, false)
  local client = llm_router.http_client({ timeout_ms = 60000 })
  for _, credential_entry in ipairs(api_keys) do
    local response, fault = client:request({ method = "POST", url = API_URL .. "/messages", headers = api_headers(credential_entry), body = json.encode(payload) })
    if fault == nil then
      tried = tried + 1
      if response.status == 200 then return { status = "healthy", response = "ok" } end
      if response.status == 429 or response.status == 402 then seen_quota = true end
      if response.status == 404 then missing = missing + 1 end
      last = "status " .. tostring(response.status)
    end
  end
  if tried > 0 and missing == tried and #subscriptions == 0 then return { status = "unhealthy", message = "model not found upstream" } end
  if seen_quota then return { status = "quota_exceeded", message = "quota exhausted" } end
  return { status = "unknown", message = last }
end

-- ── Registration ──

llm_router.register("claude", {
  icon = "https://claude.ai/favicon.ico",

  locales = {
    en = {
      ["claude.field.api_key"] = "API Key",
      ["claude.field.code"] = "Code",
      ["claude.config.credential_order"] = "Credential Order",
      ["claude.config.subscription_first"] = "Subscription First",
      ["claude.config.api_key_first"] = "API Key First",
      ["claude.sign_in.title"] = "Sign In with Claude",
      ["claude.sign_in.subtitle"] = "Claude Code signs in with your account. The sign-in stays in its container.",
      ["claude.sign_in.own_use"] = "Use only your own subscription, for yourself. Do not give other people access to this provider.",
      ["claude.sign_in.step_open"] = "1. Open the sign-in page and sign in.",
      ["claude.sign_in.open"] = "Open the sign-in page",
      ["claude.sign_in.step_code"] = "2. Paste the code the page shows.",
      ["claude.action.finish"] = "Finish",
      ["claude.action.start_over"] = "Start Over",
      ["claude.action.try_again"] = "Try Again",
      ["claude.error.preparing"] = "The bridge is still installing. Try again in a minute.",
      ["claude.error.container"] = "The bridge container is not available.",
      ["claude.error.sign_in_failed"] = "The sign-in failed.",
      ["claude.error.expired"] = "The sign-in expired. Start again.",
      ["claude.error.missing_code"] = "Paste the code from the sign-in page.",
    },
    ru = {
      ["claude.field.api_key"] = "API-ключ",
      ["claude.field.code"] = "Код",
      ["claude.config.credential_order"] = "Порядок учетных записей",
      ["claude.config.subscription_first"] = "Сначала подписка",
      ["claude.config.api_key_first"] = "Сначала API-ключ",
      ["claude.sign_in.title"] = "Вход через Claude",
      ["claude.sign_in.subtitle"] = "Claude Code входит в ваш аккаунт. Вход остается в его контейнере.",
      ["claude.sign_in.own_use"] = "Используйте только свою подписку и только для себя. Не давайте другим людям доступ к этому провайдеру.",
      ["claude.sign_in.step_open"] = "1. Откройте страницу входа и войдите.",
      ["claude.sign_in.open"] = "Открыть страницу входа",
      ["claude.sign_in.step_code"] = "2. Вставьте код со страницы.",
      ["claude.action.finish"] = "Готово",
      ["claude.action.start_over"] = "Начать заново",
      ["claude.action.try_again"] = "Повторить",
      ["claude.error.preparing"] = "Мост еще устанавливается. Попробуйте через минуту.",
      ["claude.error.container"] = "Контейнер моста недоступен.",
      ["claude.error.sign_in_failed"] = "Вход не удался.",
      ["claude.error.expired"] = "Время входа истекло. Начните заново.",
      ["claude.error.missing_code"] = "Вставьте код со страницы входа.",
    },
  },

  config_schema = {
    {
      type = "select", name = "credential_order", label = "claude.config.credential_order",
      options = { "subscription_first", "api_key_first" },
      option_labels = { subscription_first = "claude.config.subscription_first", api_key_first = "claude.config.api_key_first" },
    },
  },

  credential_schema = {
    { type = "secret", name = "api_key", label = "claude.field.api_key", required = true },
  },

  validate_credentials = function(data)
    if type(data.account) == "string" and data.account ~= "" then
      if not string.match(data.account, "^[a-z0-9][a-z0-9%-]*$") then
        return false, { message = "account must be a bridge account slot", code = "invalid_request_error" }
      end
      return true
    end
    local api_key = trim(data.api_key)
    if api_key == "" then return false, { message = "set the API key", code = "invalid_request_error" } end
    if string.sub(api_key, 1, 7) ~= "sk-ant-" then
      return false, { message = "an Anthropic API key starts with sk-ant-", code = "invalid_request_error" }
    end
    return true
  end,

  jobs = {
    -- Brings the bridge back after a router restart while subscription
    -- accounts exist.
    bridge = {
      interval_seconds = 300,
      run_on_startup = true,
      timeout_ms = 120000,
      run = function(ctx)
        local subscriptions = ordered_credentials(ctx)
        if #subscriptions == 0 then return true end
        local fault = ensure_running(90000)
        if fault ~= nil and fault.reason ~= "preparing" then return fault end
        return true
      end,
    },
  },

  auth_initiate = function(ctx) return start_subscription(ctx) end,

  auth_step = function(ctx, input)
    local action = input.action or ""
    if action == "submit_code" then return submit_code(ctx, input.values or {}) end
    if action == "cancel" then
      abandon_flow(ctx)
      return retry_page()
    end
    return start_subscription(ctx)
  end,

  check_health = function(ctx, credential)
    local kind = kind_of(credential)
    if kind == "subscription" then return subscription_health(credential) end
    if kind == "api_key" then return api_health(credential) end
    return { status = "unhealthy", message = "the credential has neither an API key nor an account" }
  end,

  check_model = function(ctx, model_ref)
    if model_ref.endpoint ~= "chat" and model_ref.endpoint ~= "vision" then
      return { status = "unknown", message = "endpoint not supported: " .. tostring(model_ref.endpoint) }
    end
    return probe_model(ctx, model_ref)
  end,

  get_model_infos = function(ctx)
    local cards, seen = {}, {}
    local _, api_keys = ordered_credentials(ctx)
    for _, credential_entry in ipairs(api_keys) do
      local listed, fault = api_models(credential_entry)
      if fault == nil then
        for _, card in ipairs(listed) do
          seen[card.name] = true
          table.insert(cards, card)
        end
        break
      end
      llm_router.log("warn", "model list failed", { cred_id = credential_entry.id, error = fault.message })
    end
    for _, model in ipairs(SUBSCRIPTION_MODELS) do
      if not seen[model.name] then table.insert(cards, model_card(model.name, model.display_name, 1000000, 128000)) end
    end
    return cards, nil
  end,

  complete = function(ctx, request)
    return serve(ctx,
      function(credentials) return subscription_complete(ctx, request, credentials) end,
      function(credentials) return api_complete(ctx, request, credentials) end)
  end,

  complete_stream = function(ctx, request, emit)
    local _, failure = serve(ctx,
      function(credentials)
        local stream_failure, fall_through = subscription_stream(ctx, request, credentials, emit)
        return nil, stream_failure, fall_through
      end,
      function(credentials)
        local stream_failure, fall_through = api_stream(ctx, request, credentials, emit)
        return nil, stream_failure, fall_through
      end)
    return nil, failure
  end,
})
