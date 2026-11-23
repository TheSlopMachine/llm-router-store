--- @plugin Mock OpenAI Server
--- @author TheSlopMachine
--- @version 0.1.0
--- @plugin_api 1.8
--- @description Mock OpenAI API in a Docker container: chat, streaming, embeddings, images, speech, transcription
--- @license MIT
--- @allow_host codeload.github.com
--- @require_api docker
--- @docker.build mock https://codeload.github.com/freakynit/mock-openai-server/tar.gz/20d1a0fac8c56b39e512a74d44f9d2bdf8363d83
--- @docker.memory_limit_mb 512
--- @docker.cpu_limit 1
--- @docker.container_limit 1

-- freakynit/mock-openai-server built from a pinned commit. The container
-- listens on 8080 and accepts the API keys of its bundled config.yaml.

local docker = llm_router.docker

local KEY = "mock"
local PORT = 8080
local API_KEY = "key-1"

local SPEC = {
  build = "mock",
  ports = { PORT },
  restart = "on_failure",
  health = {
    command = { "node", "-e", "require('net').connect(8080,'127.0.0.1').on('connect',()=>process.exit(0)).on('error',()=>process.exit(1))" },
    interval_seconds = 5,
    timeout_seconds = 3,
    start_period_seconds = 10,
    retries = 6,
  },
}

-- Endpoints per model of the bundled config. Unknown models serve chat.
local MODEL_ENDPOINTS = {
  ["dall-e-2"] = { "images/generations" },
  ["dall-e-3"] = { "images/generations" },
  ["tts-1"] = { "audio/speech" },
  ["tts-1-hd"] = { "audio/speech" },
  ["whisper-1"] = { "audio/transcriptions", "audio/translations" },
  ["text-embedding-ada-002"] = { "embeddings" },
}

local function not_ready(message)
  return { message = message, code = "server_error", status = 503 }
end

-- Builds or recreates the container when needed, starts it and waits until
-- the port answers. Every step is idempotent.
local function ensure_running(timeout_ms)
  local status, fault = docker.ensure(KEY, SPEC)
  if fault ~= nil then return fault end
  if status.state == "preparing" then
    return not_ready("mock server image is building")
  end
  local _, start_fault = docker.start(KEY)
  if start_fault ~= nil then return start_fault end
  local _, wait_fault = docker.wait(KEY, { port = PORT, timeout_ms = timeout_ms or 30000 })
  return wait_fault
end

local function stopped(fault)
  return fault ~= nil and fault.docker and (fault.reason == "not_running" or fault.reason == "not_found")
end

local function with_auth(arguments)
  arguments.headers = arguments.headers or {}
  arguments.headers["authorization"] = "Bearer " .. API_KEY
  return arguments
end

-- One unary call. A stopped or missing container starts once, then the
-- call repeats.
local function call(arguments, timeout_ms)
  local client = docker.open_http(KEY, PORT, { timeout_ms = timeout_ms or 120000 })
  with_auth(arguments)
  local response, fault = client:request(arguments)
  if stopped(fault) then
    local start_fault = ensure_running()
    if start_fault ~= nil then return nil, start_fault end
    response, fault = client:request(arguments)
  end
  return response, fault
end

local function upstream_error(status, body)
  local code = "server_error"
  if status == 400 or status == 404 then code = "invalid_request_error" end
  return { message = "mock server status " .. tostring(status) .. ": " .. (body or ""), code = code, status = status >= 400 and status <= 599 and status or 502 }
end

local function post_json(path, payload)
  local response, fault = call({
    method = "POST", url = path,
    headers = { ["content-type"] = "application/json" },
    body = json.encode(payload),
  })
  if fault ~= nil then return nil, fault end
  if response.status ~= 200 then return nil, upstream_error(response.status, response.body) end
  return response, nil
end

local function chat_payload(request, stream)
  return {
    model = request.model_name,
    messages = request.messages,
    tools = request.tools,
    tool_choice = request.tool_choice,
    max_tokens = request.max_tokens,
    max_completion_tokens = request.max_completion_tokens,
    temperature = request.temperature,
    top_p = request.top_p,
    stop = request.stop,
    response_format = request.response_format,
    stream = stream,
    stream_options = stream and { include_usage = true } or nil,
  }
end

llm_router.register("mock-openai-server", {
  initialize = function(ctx)
    local fault = ensure_running(20000)
    if fault ~= nil then
      llm_router.log("info", "mock server not ready at initialization", { reason = fault.reason, error = fault.message })
    end
  end,

  jobs = {
    keep_running = {
      interval_seconds = 60,
      run_on_startup = true,
      timeout_ms = 120000,
      run = function(ctx)
        local fault = ensure_running(90000)
        if fault ~= nil and fault.reason ~= "preparing" then
          llm_router.log("warn", "mock server not running", { reason = fault.reason, error = fault.message })
        end
        return true
      end,
    },
  },

  complete = function(ctx, request)
    local response, fault = post_json("/v1/chat/completions", chat_payload(request, false))
    if fault ~= nil then return nil, fault end
    return json.decode(response.body), nil
  end,

  complete_stream = function(ctx, request, emit)
    local client = docker.open_http(KEY, PORT, { timeout_ms = 900000 })
    local arguments = with_auth({
      method = "POST", url = "/v1/chat/completions",
      headers = { ["content-type"] = "application/json" },
      body = json.encode(chat_payload(request, true)),
      on_response = function(head)
        if head.status ~= 200 then return upstream_error(head.status, head.body) end
      end,
      on_line = function(line)
        local payload = line:match("^data: (.*)$")
        if payload == nil or payload == "[DONE]" then return end
        local chunk = json.decode(payload)
        if (chunk.choices ~= nil and #chunk.choices > 0) or chunk.usage ~= nil then emit(chunk) end
      end,
    })
    local _, fault = client:stream(arguments)
    if stopped(fault) then
      local start_fault = ensure_running()
      if start_fault ~= nil then return nil, start_fault end
      _, fault = client:stream(arguments)
    end
    return nil, fault
  end,

  embed = function(ctx, request)
    local response, fault = post_json("/v1/embeddings", {
      model = request.model_name, input = request.input, encoding_format = "float", dimensions = request.dimensions,
    })
    if fault ~= nil then return nil, fault end
    return json.decode(response.body), nil
  end,

  generate_image = function(ctx, request)
    local response, fault = post_json("/v1/images/generations", {
      model = request.model_name, prompt = request.prompt, n = request.n, size = request.size,
      quality = request.quality, style = request.style, response_format = "b64_json",
    })
    if fault ~= nil then return nil, fault end
    local payload = json.decode(response.body)
    return { created = payload.created, data = payload.data }, nil
  end,

  speech = function(ctx, request)
    local format = request.response_format or "mp3"
    local response, fault = post_json("/v1/audio/speech", {
      model = request.model_name, input = request.input, voice = request.voice or "alloy",
      response_format = format, speed = request.speed,
    })
    if fault ~= nil then return nil, fault end
    return { audio_b64 = llm_router.base64_encode(response.body), format = format }, nil
  end,

  transcribe = function(ctx, request)
    local path = "/v1/audio/transcriptions"
    if request.task == "translate" then path = "/v1/audio/translations" end
    local parts = {
      { name = "file", data = request.file, filename = request.file_name or "audio.wav", content_type = request.content_type },
      { name = "model", value = request.model_name },
      { name = "response_format", value = "json" },
    }
    if request.language ~= nil and request.task ~= "translate" then
      table.insert(parts, { name = "language", value = request.language })
    end
    local body, content_type = llm_router.multipart(parts)
    local response, fault = call({ method = "POST", url = path, headers = { ["content-type"] = content_type }, body = body })
    if fault ~= nil then return nil, fault end
    if response.status ~= 200 then return nil, upstream_error(response.status, response.body) end
    local payload = json.decode(response.body)
    return { text = payload.text, task = request.task }, nil
  end,

  get_model_infos = function(ctx)
    local response, fault = call({ method = "GET", url = "/v1/models" }, 30000)
    if fault ~= nil then return nil, fault end
    if response.status ~= 200 then return nil, upstream_error(response.status, response.body) end
    local cards = {}
    for _, entry in ipairs(json.decode(response.body).data or {}) do
      table.insert(cards, {
        name = entry.id,
        display_name = entry.id,
        endpoints = MODEL_ENDPOINTS[entry.id] or { "chat/completions" },
      })
    end
    return cards, nil
  end,

  check_model = function(ctx, model_ref)
    if model_ref.endpoint ~= "chat" then
      local response, fault = call({ method = "GET", url = "/v1/models/" .. model_ref.model_name }, 30000)
      if fault ~= nil then return { status = "unknown", message = fault.message } end
      if response.status == 200 then return { status = "healthy", response = "listed" } end
      if response.status == 404 then return { status = "unhealthy", message = "model not configured" } end
      return { status = "unknown", message = "status " .. tostring(response.status) }
    end
    local response, fault = post_json("/v1/chat/completions", {
      model = model_ref.model_name, messages = { { role = "user", content = "hi" } }, max_tokens = 8,
    })
    if fault ~= nil then
      if fault.status == 400 then return { status = "unhealthy", message = fault.message } end
      return { status = "unknown", message = fault.message }
    end
    local payload = json.decode(response.body)
    local content = payload.choices and payload.choices[1] and payload.choices[1].message and payload.choices[1].message.content or ""
    return { status = "healthy", response = string.sub(content, 1, 60) }
  end,
})
