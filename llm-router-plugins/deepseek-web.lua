--- @plugin DeepSeek Web
--- @author TheSlopMachine
--- @version 0.3.1
--- @plugin_api 1.8
--- @description DeepSeek web chat through ds-web-server in a Docker container: thinking, streaming, tools, files, web search
--- @license MIT
--- @allow_host chat.deepseek.com
--- @allow_host fe-static.deepseek.com
--- @allow_host hif-leim.deepseek.com
--- @allow_host hif-dliq.deepseek.com
--- @require_api docker
--- @docker.image ghcr.io/theslopmachine/ds-free-api:v0.6.0@sha256:ec6a23a4e50961e1eb929efefa68d9facb05a286926c4bdbd2aa2393dccb8887
--- @docker.memory_limit_mb 512
--- @docker.cpu_limit 1
--- @docker.container_limit 1

-- ds-web-server (github.com/TheSlopMachine/ds-web-server, a ds-free-api
-- fork) serves the DeepSeek web chat as an OpenAI API. One container holds
-- every DeepSeek account of the provider. The router owns the accounts:
-- they are its credentials. The sync job pushes them into the container
-- account pool, and each request names its account in X-DS-Account, so a
-- failure parks or disables exactly that credential.

local docker = llm_router.docker

local IMAGE = "ghcr.io/theslopmachine/ds-free-api:v0.6.0@sha256:ec6a23a4e50961e1eb929efefa68d9facb05a286926c4bdbd2aa2393dccb8887"
local KEY = "deepseek"
local PORT = 22217
local CONFIG_PATH = "/app/config/config.toml"
local SEARCH_SUFFIX = "-search"
local SIGN_IN_URL = "https://chat.deepseek.com/sign_in"
local DEVICE_ID_COMMAND = "copy(SMSdk.getDeviceId())"
local DEFAULT_AREA_CODE = "+86"
local STATE_SCOPE = "container"
local ACCOUNT_FIELDS = { "email", "mobile", "area_code", "password", "device_id" }
-- Parking time when the container gives no Retry-After.
local DEFAULT_PARK_SECONDS = 60
-- A dead account (wrong password, ban) waits this long when the provider
-- does not allow disabling credentials.
local DEAD_PARK_SECONDS = 3600
-- The router parks for at most 90 days; a mute can last longer.
local MAX_PARK_SECONDS = 90 * 24 * 3600

-- The Android app identity: from a European address the CloudFront WAF
-- challenges the web identity at sign-in and lets the app identity through.
-- The container reads the gateway proxy from HTTPS_PROXY. Search runs only
-- on request.
local CORE_SETTINGS = {
  { "emulation", "okhttp4_12" },
  { "user_agent", "DeepSeek/2.5.0 Android/35" },
  { "client_platform", "android" },
  { "client_os", "android" },
  { "default_search_enabled", false },
}

-- ── Container ──

local function state_secrets()
  local stored = llm_router.storage.get(STATE_SCOPE, "secrets") or {}
  if stored.api_key == nil or stored.admin_token == nil then
    stored = {
      api_key = stored.api_key or ("sk-" .. llm_router.random_hex(24)),
      admin_token = llm_router.random_hex(32),
    }
    llm_router.storage.set(STATE_SCOPE, "secrets", stored)
  end
  return stored
end

local function toml_value(value)
  if type(value) == "boolean" then return tostring(value) end
  return "\"" .. value .. "\""
end

local function config_toml(secrets)
  local lines = {
    "[server]", "host = \"0.0.0.0\"", "port = " .. PORT, "",
    "[[api_keys]]", "key = \"" .. secrets.api_key .. "\"", "description = \"llm-router\"", "",
    "[admin]", "api_token = \"" .. secrets.admin_token .. "\"", "",
    "[ds_core]",
  }
  for _, setting in ipairs(CORE_SETTINGS) do
    table.insert(lines, setting[1] .. " = " .. toml_value(setting[2]))
  end
  return table.concat(lines, "\n") .. "\n"
end

local function container_spec()
  return {
    image = IMAGE,
    ports = { PORT },
    files = { [CONFIG_PATH] = config_toml(state_secrets()) },
    restart = "on_failure",
    -- The port opens before the server answers. "-Y off": the health probe
    -- must not go through the gateway proxy.
    health = {
      command = { "wget", "-q", "-Y", "off", "-O", "/dev/null", "http://127.0.0.1:" .. PORT .. "/health" },
      interval_seconds = 2,
      timeout_seconds = 2,
      start_period_seconds = 2,
      retries = 15,
    },
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
      message = "the ds-web-server image is downloading, try again in a minute",
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

local function container_request(arguments, timeout_ms)
  local client = docker.open_http(KEY, PORT, { timeout_ms = timeout_ms or 120000 })
  return client:request(arguments)
end

-- Runs one call against the container and starts a stopped container once.
-- A fresh container pool fills on demand: requests sync their account.
local function with_container(run)
  local result, fault = run()
  if not stopped(fault) then return result, fault end
  local start_fault = ensure_running()
  if start_fault ~= nil then return nil, start_fault end
  return run()
end

-- The container answers {"error": "text"} on the admin API and
-- {"error": {"message", "code"}} on /v1.
local function decode_error(body)
  local ok, decoded = pcall(json.decode, body or "")
  if ok and type(decoded) == "table" then
    if type(decoded.error) == "table" then return decoded.error.message or "", decoded.error.code end
    if type(decoded.error) == "string" then return decoded.error, nil end
  end
  return string.sub(body or "", 1, 1000), nil
end

-- ── Accounts ──

local function account_of(data)
  local account = {}
  for _, field in ipairs(ACCOUNT_FIELDS) do account[field] = (data or {})[field] or "" end
  return account
end

local function account_id(account)
  if account.email ~= "" then return account.email end
  return account.mobile
end

local function path_escape(value)
  return (string.gsub(value, "[^%w%-%._~@]", function(c) return string.format("%%%02X", string.byte(c)) end))
end

local function admin_call(method, path, body, timeout_ms)
  local arguments = {
    method = method, url = path,
    headers = { authorization = "Bearer " .. state_secrets().admin_token },
  }
  if body ~= nil then
    arguments.headers["content-type"] = "application/json"
    arguments.body = body
  end
  local response, fault = with_container(function() return container_request(arguments, timeout_ms or 30000) end)
  if fault ~= nil then return nil, fault end
  if response.status ~= 200 then
    local message = decode_error(response.body)
    return nil, {
      message = "ds-web-server " .. method .. " " .. path .. " answered " .. tostring(response.status) .. ": " .. message,
      code = "server_error", status = 502,
    }
  end
  return json.decode(response.body), nil
end

-- Makes the container pool hold exactly these accounts. Unchanged accounts
-- keep their session; new and changed ones sign in before the call returns.
local function replace_accounts(accounts)
  local body = '{"accounts":[]}'
  if #accounts > 0 then body = json.encode({ accounts = accounts }) end
  return admin_call("PUT", "/admin/api/accounts", body, 900000)
end

-- Adds or updates one account and returns its pool status.
local function put_account(account)
  return admin_call("PUT", "/admin/api/accounts/" .. path_escape(account_id(account)), json.encode(account), 300000)
end

local function delete_account(id)
  return admin_call("DELETE", "/admin/api/accounts/" .. path_escape(id))
end

local function account_status(id)
  local listing, fault = admin_call("GET", "/admin/api/accounts")
  if fault ~= nil then return nil, fault end
  for _, entry in ipairs(listing.accounts or {}) do
    if entry.id == id then return entry, nil end
  end
  return nil, nil
end

local function accounts_of(credential_entries)
  local accounts = {}
  for _, credential_entry in ipairs(credential_entries) do
    local account = account_of(credential_entry.data)
    if account_id(account) ~= "" then table.insert(accounts, account) end
  end
  return accounts
end

-- ── Credential outcomes ──

-- Error codes of a request pinned with X-DS-Account. "park": transient, the
-- credential waits Retry-After. "dead": DeepSeek rejects the account itself.
local ACCOUNT_OUTCOMES = {
  account_not_found = "sync",
  account_busy = "park",
  account_quota_exhausted = "park",
  account_muted = "park",
  account_recovering = "park",
  account_unavailable = "park",
  account_blocked = "park",
  account_invalid_credentials = "dead",
  account_banned = "dead",
  account_device_rejected = "dead",
  account_rejected = "dead",
  account_health_check_failed = "dead",
  account_invalid_config = "dead",
}

local function park(credential_entry, seconds, reason)
  pcall(llm_router.credentials.park, credential_entry.id, seconds, reason)
  llm_router.log("warn", "credential parked", { cred_id = credential_entry.id, wait = seconds, reason = reason })
end

local function retire(ctx, credential_entry, reason)
  local config = ctx.provider_config
  if type(config) == "table" and config.disable_failed_credentials == true
    and pcall(llm_router.credentials.disable, credential_entry.id, reason) then
    llm_router.log("warn", "credential disabled", { cred_id = credential_entry.id, reason = reason })
    return
  end
  park(credential_entry, DEAD_PARK_SECONDS, reason)
end

-- ── Chat ──

local function api_headers(account)
  return {
    ["content-type"] = "application/json",
    authorization = "Bearer " .. state_secrets().api_key,
    ["x-ds-account"] = account_id(account),
  }
end

-- "<model>-search" is the same model with DeepSeek web search on.
local function upstream_model(model_name)
  if #model_name > #SEARCH_SUFFIX and string.sub(model_name, -#SEARCH_SUFFIX) == SEARCH_SUFFIX then
    return string.sub(model_name, 1, #model_name - #SEARCH_SUFFIX), true
  end
  return model_name, false
end

-- Thinking is on unless reasoning_effort is "none". Every other effort
-- turns it on: DeepSeek has no effort levels.
local function chat_payload(request, stream)
  local model, search = upstream_model(request.model_name)
  local payload = {
    model = model,
    messages = request.messages,
    tools = request.tools,
    tool_choice = request.tool_choice,
    parallel_tool_calls = request.parallel_tool_calls,
    max_tokens = request.max_tokens,
    max_completion_tokens = request.max_completion_tokens,
    stop = request.stop,
    response_format = request.response_format,
    reasoning_effort = request.reasoning_effort,
    stream = stream,
  }
  if stream then payload.stream_options = { include_usage = true } end
  if search then payload.web_search_options = { search_context_size = "high" } end
  return payload
end

local function upstream_error(status, body, retry_after)
  local text = decode_error(body)
  local message = "ds-web-server status " .. tostring(status) .. ": " .. text
  if status == 429 then
    local result = { message = message, code = "rate_limit", status = 429 }
    if retry_after ~= nil then result.headers = { ["Retry-After"] = retry_after } end
    return result
  end
  if status == 400 then
    -- ds-web-server names an unknown model with this text.
    if string.find(text, "不支持的模型", 1, true) ~= nil then
      return { message = message, code = "not_found", status = 404 }
    end
    return { message = message, code = "invalid_request_error", status = 400 }
  end
  return { message = message, code = "upstream_error", status = 502 }
end

-- What to do with a non-200 answer: nil when it is not about the account,
-- otherwise the outcome name from ACCOUNT_OUTCOMES, the code and the wait.
local function account_outcome(head)
  local _, code = decode_error(head.body)
  return ACCOUNT_OUTCOMES[code or ""], code, tonumber(head.headers["retry-after"])
end

-- The same decision for an account that failed to sign in during a sync.
local function failure_outcome(failure)
  local code = "account_" .. (failure.kind or "rejected")
  return ACCOUNT_OUTCOMES[code] or "dead", code, nil
end

-- Runs one chat call per usable credential until one answers.
--   attempt(credential_entry, account) -> response_head, result, fault
-- response_head is {status, headers, body} for a non-200 answer.
local function each_credential(ctx, attempt)
  local tried = 0
  for _, credential_entry in ipairs(llm_router.credentials.list()) do
    local account = account_of(credential_entry.data)
    if account_id(account) ~= "" then
      tried = tried + 1
      local head, result, fault = attempt(credential_entry, account)
      local outcome, code, wait = nil, nil, nil
      if head ~= nil then outcome, code, wait = account_outcome(head) end
      if outcome == "sync" then
        -- The container lost the account (a fresh container): add it and retry.
        local status, sync_fault = put_account(account)
        if sync_fault ~= nil then return nil, sync_fault end
        if status.state == "idle" or status.state == "busy" then
          head, result, fault = attempt(credential_entry, account)
          outcome = nil
          if head ~= nil then outcome, code, wait = account_outcome(head) end
          -- Lost again right after the sync: wait, do not loop.
          if outcome == "sync" then outcome = "park" end
        else
          outcome, code, wait = failure_outcome(status.failure or {})
        end
      end
      if outcome == nil then
        if head ~= nil then return nil, upstream_error(head.status, head.body, head.headers["retry-after"]) end
        return result, fault
      end
      if outcome == "park" then
        park(credential_entry, math.min(math.max(wait or DEFAULT_PARK_SECONDS, 1), MAX_PARK_SECONDS), code)
      else
        retire(ctx, credential_entry, code)
      end
    end
  end
  if tried == 0 then
    return nil, { message = "no DeepSeek account is signed in", code = "auth_error", status = 401 }
  end
  return nil, { message = "no DeepSeek account accepted the request", code = "rate_limit", status = 429 }
end

local function post_chat(ctx, payload, timeout_ms)
  return each_credential(ctx, function(_, account)
    local response, fault = with_container(function()
      return container_request({
        method = "POST", url = "/v1/chat/completions", headers = api_headers(account), body = json.encode(payload),
      }, timeout_ms or 900000)
    end)
    if fault ~= nil then return nil, nil, fault end
    if response.status ~= 200 then return response, nil, nil end
    return nil, json.decode(response.body), nil
  end)
end

local function stream_chat(ctx, payload, emit)
  local body = json.encode(payload)
  return each_credential(ctx, function(_, account)
    local rejected = nil
    local client = docker.open_http(KEY, PORT, { timeout_ms = 900000 })
    local _, fault = with_container(function()
      return client:stream({
        method = "POST", url = "/v1/chat/completions", headers = api_headers(account), body = body,
        on_response = function(head)
          if head.status ~= 200 then
            rejected = head
            return upstream_error(head.status, head.body, head.headers["retry-after"])
          end
        end,
        on_line = function(line)
          local data = string.match(line, "^data: (.*)$")
          if data == nil or data == "[DONE]" then return end
          local chunk = json.decode(data)
          if (chunk.choices ~= nil and #chunk.choices > 0) or chunk.usage ~= nil then emit(chunk) end
        end,
      })
    end)
    if rejected ~= nil then return rejected, nil, nil end
    return nil, nil, fault
  end)
end

-- ── Model cards ──

local function display_name(model)
  local kind = string.gsub(model, "^deepseek%-", "")
  return "DeepSeek " .. string.upper(string.sub(kind, 1, 1)) .. string.sub(kind, 2)
end

local function model_card(entry, search)
  local card = {
    name = entry.id,
    display_name = display_name(entry.id),
    description = "DeepSeek web chat.",
    context_window = entry.context_window,
    max_tokens = entry.max_tokens,
    input_modalities = { "text", "image", "file" },
    output_modalities = { "text" },
    capabilities = { "tools", "reasoning" },
    reasoning = { supported_efforts = { "high", "none" }, default_effort = "high", default_enabled = true },
    endpoints = { "chat/completions" },
  }
  if search then
    card.name = entry.id .. SEARCH_SUFFIX
    card.display_name = card.display_name .. " Search"
    card.description = "DeepSeek web chat with web search."
  end
  return card
end

-- ── Sign-in wizard ──

local function trim(value)
  if type(value) ~= "string" then return "" end
  return (string.gsub(value, "^%s*(.-)%s*$", "%1"))
end

-- The console prints the ID in quotes. Users paste either form.
local function clean_device_id(value)
  local text = trim(value)
  local unquoted = string.match(text, "^[\"'](.*)[\"']$")
  return unquoted or text
end

local function account_from_form(values)
  local login = trim(values.login)
  local password = values.password
  local device_id = clean_device_id(values.device_id)
  if login == "" or type(password) ~= "string" or password == "" or device_id == "" then
    return nil, "deepseek.error.missing_fields"
  end
  local account = { email = "", mobile = "", area_code = "", password = password, device_id = device_id }
  if string.find(login, "@", 1, true) ~= nil then
    account.email = login
  else
    account.mobile = (string.gsub(login, "[%s%-]", ""))
    account.area_code = trim(values.area_code)
    if account.area_code == "" then account.area_code = DEFAULT_AREA_CODE end
  end
  return account, nil
end

-- Sign-in failure kinds with their own explanation.
local FAILURE_KEYS = {
  invalid_credentials = "deepseek.error.invalid_credentials",
  device_rejected = "deepseek.error.device_rejected",
  muted = "deepseek.error.muted",
  banned = "deepseek.error.banned",
  blocked = "deepseek.error.blocked",
  unavailable = "deepseek.error.unavailable",
}

local function sign_in_page(values, error_key, detail)
  values = values or {}
  local nodes = {}
  if error_key ~= nil then
    table.insert(nodes, { type = "banner", variant = "error", text = error_key })
    if detail ~= nil and detail ~= "" then table.insert(nodes, { type = "code", text = detail }) end
  end
  table.insert(nodes, {
    type = "section",
    title = "deepseek.sign_in.title",
    subtitle = "deepseek.sign_in.subtitle",
    content = {
      { type = "text", text = "deepseek.sign_in.step_browser" },
      { type = "link", text = "deepseek.sign_in.open", url = SIGN_IN_URL },
      { type = "text", text = "deepseek.sign_in.step_console" },
      { type = "code", text = DEVICE_ID_COMMAND },
      {
        type = "grid", columns = 2,
        content = {
          { type = "input", name = "login", label = "deepseek.field.login", placeholder = "deepseek.field.login_placeholder", value = values.login, required = true },
          { type = "input", name = "area_code", label = "deepseek.field.area_code", placeholder = DEFAULT_AREA_CODE, value = values.area_code },
        },
      },
      { type = "secret", name = "password", label = "deepseek.field.password", value = values.password, required = true },
      { type = "input", name = "device_id", label = "deepseek.field.device_id", value = values.device_id, required = true },
      { type = "button", text = "deepseek.sign_in.submit", form_action = "sign_in", variant = "primary" },
    },
  })
  return { render = nodes }
end

-- ── Registration ──

llm_router.register("deepseek-web", {
  icon = "https://www.deepseek.com/favicon.ico",

  credential_schema = {
    { type = "input", name = "email", label = "deepseek.field.email" },
    { type = "input", name = "mobile", label = "deepseek.field.mobile" },
    { type = "input", name = "area_code", label = "deepseek.field.area_code", placeholder = DEFAULT_AREA_CODE },
    { type = "secret", name = "password", label = "deepseek.field.password", required = true },
    { type = "input", name = "device_id", label = "deepseek.field.device_id", required = true },
  },

  validate_credentials = function(data)
    local account = account_of(data)
    if account_id(account) == "" then return false, { message = "set an email or a phone number" } end
    if account.password == "" then return false, { message = "set the password" } end
    if account.device_id == "" then return false, { message = "set the device ID" } end
    return true
  end,

  initialize = function(ctx)
    local fault = ensure_running(20000)
    if fault ~= nil then
      llm_router.log("info", "container not ready at initialization", { reason = fault.reason, error = fault.message })
    end
  end,

  jobs = {
    sync = {
      interval_seconds = 60,
      run_on_startup = true,
      timeout_ms = 900000,
      run = function(ctx)
        local fault = ensure_running(120000)
        if fault ~= nil then
          if fault.reason == "preparing" then return true end
          return fault
        end
        local accounts = accounts_of(llm_router.credentials.list())
        local listing, sync_fault = replace_accounts(accounts)
        if sync_fault ~= nil then return sync_fault end
        for _, entry in ipairs(listing.accounts or {}) do
          if entry.failure ~= nil then
            llm_router.log("warn", "account not signed in", { reason = entry.failure.kind, error = entry.failure.message })
          end
        end
        return true
      end,
    },
  },

  auth_initiate = function(ctx) return sign_in_page() end,

  auth_step = function(ctx, input)
    local values = input.values or {}
    if input.action ~= "sign_in" then return sign_in_page(values) end
    local account, problem = account_from_form(values)
    if problem ~= nil then return sign_in_page(values, problem) end

    local fault = ensure_running(90000)
    if fault ~= nil then
      if fault.reason == "preparing" then return sign_in_page(values, "deepseek.error.preparing") end
      return sign_in_page(values, "deepseek.error.container", fault.message)
    end
    local status, put_fault = put_account(account)
    if put_fault ~= nil then return sign_in_page(values, "deepseek.error.container", put_fault.message) end
    if status.state == "idle" or status.state == "busy" then
      llm_router.log("info", "account signed in", { operation = "auth_step" })
      return { credentials = account }
    end

    local failure = status.failure or {}
    -- A failed account leaves the pool: the pool holds router credentials only.
    local _, drop_fault = delete_account(account_id(account))
    if drop_fault ~= nil then
      llm_router.log("warn", "failed account stays in the pool", { error = drop_fault.message })
    end
    llm_router.log("warn", "account sign-in failed", { reason = failure.kind, error = failure.message, operation = "auth_step" })
    return sign_in_page(values, FAILURE_KEYS[failure.kind or ""] or "deepseek.error.sign_in_failed", failure.message)
  end,

  check_health = function(ctx, credential)
    local account = account_of(credential.data)
    local id = account_id(account)
    local fault = ensure_running(60000)
    if fault ~= nil then return { status = "unknown", message = fault.message } end
    local status, status_fault = account_status(id)
    if status_fault ~= nil then return { status = "unknown", message = status_fault.message } end
    if status == nil then
      status, status_fault = put_account(account)
      if status_fault ~= nil then return { status = "unknown", message = status_fault.message } end
    end
    if status.state == "idle" or status.state == "busy" then return { status = "healthy" } end
    local failure = status.failure or {}
    if status.state == "error" or failure.kind == "unavailable" then
      return { status = "unknown", message = "sign-in is being retried: " .. (failure.message or "no detail") }
    end
    if failure.kind == "muted" then return { status = "unknown", message = "muted: " .. (failure.message or "no detail") } end
    return { status = "unhealthy", message = (failure.kind or "invalid") .. ": " .. (failure.message or "no detail") }
  end,

  complete = function(ctx, request)
    return post_chat(ctx, chat_payload(request, false))
  end,

  complete_stream = function(ctx, request, emit)
    local _, fault = stream_chat(ctx, chat_payload(request, true), emit)
    return nil, fault
  end,

  get_model_infos = function(ctx)
    local response, fault = with_container(function()
      return container_request({
        method = "GET", url = "/v1/models",
        headers = { authorization = "Bearer " .. state_secrets().api_key },
      }, 30000)
    end)
    if fault ~= nil then return nil, fault end
    if response.status ~= 200 then return nil, upstream_error(response.status, response.body) end
    local cards = {}
    -- The list repeats each model under its bare name: keep the deepseek- ids.
    for _, entry in ipairs(json.decode(response.body).data or {}) do
      if string.find(entry.id, "^deepseek%-") ~= nil then
        table.insert(cards, model_card(entry, false))
        table.insert(cards, model_card(entry, true))
      end
    end
    return cards, nil
  end,

  check_model = function(ctx, model_ref)
    if model_ref.endpoint ~= "chat" and model_ref.endpoint ~= "vision" then
      return { status = "unknown", message = "endpoint not supported " .. tostring(model_ref.endpoint) }
    end
    local result, fault = post_chat(ctx, chat_payload({
      model_name = model_ref.model_name,
      messages = { { role = "user", content = "hi" } },
      max_tokens = 8,
      reasoning_effort = "none",
    }, false), 120000)
    if fault ~= nil then
      if fault.code == "rate_limit" then return { status = "quota_exceeded", message = fault.message } end
      if fault.code == "not_found" or fault.code == "upstream_error" then return { status = "unhealthy", message = fault.message } end
      return { status = "unknown", message = fault.message }
    end
    local message = ((result.choices or {})[1] or {}).message or {}
    return { status = "healthy", response = string.sub(message.content or "", 1, 60) }
  end,

  locales = {
    en = {
      ["deepseek.sign_in.title"] = "Sign In to DeepSeek",
      ["deepseek.sign_in.subtitle"] = "The router stores the account and keeps the container in sync.",
      ["deepseek.sign_in.step_browser"] = "1. Sign in at chat.deepseek.com in Chrome, Edge or Firefox. Use a separate browser profile for each account.",
      ["deepseek.sign_in.open"] = "Open DeepSeek",
      ["deepseek.sign_in.step_console"] = "2. On that page press F12, open Console and run this command. It copies the device ID. If the console blocks pasting, type allow pasting first.",
      ["deepseek.sign_in.submit"] = "Sign In",
      ["deepseek.field.login"] = "Email or Phone",
      ["deepseek.field.login_placeholder"] = "you@example.com",
      ["deepseek.field.email"] = "Email",
      ["deepseek.field.mobile"] = "Phone",
      ["deepseek.field.area_code"] = "Area Code",
      ["deepseek.field.password"] = "Password",
      ["deepseek.field.device_id"] = "Device ID",
      ["deepseek.error.missing_fields"] = "Enter the email or phone, the password and the device ID.",
      ["deepseek.error.preparing"] = "The ds-web-server image is downloading. Try again in a minute.",
      ["deepseek.error.container"] = "The ds-web-server container is not ready.",
      ["deepseek.error.sign_in_failed"] = "DeepSeek rejected the sign-in.",
      ["deepseek.error.invalid_credentials"] = "Wrong email, phone or password.",
      ["deepseek.error.device_rejected"] = "DeepSeek rejected the device ID. Copy it again from the browser where you signed in.",
      ["deepseek.error.muted"] = "DeepSeek muted this account. Try again after the mute ends.",
      ["deepseek.error.banned"] = "DeepSeek banned this account.",
      ["deepseek.error.blocked"] = "DeepSeek blocked the sign-in from this network.",
      ["deepseek.error.unavailable"] = "DeepSeek did not answer. Try again.",
    },
    ru = {
      ["deepseek.sign_in.title"] = "Вход в DeepSeek",
      ["deepseek.sign_in.subtitle"] = "Роутер хранит аккаунт и синхронизирует его с контейнером.",
      ["deepseek.sign_in.step_browser"] = "1. Войдите на chat.deepseek.com в Chrome, Edge или Firefox. Для каждого аккаунта используйте отдельный профиль браузера.",
      ["deepseek.sign_in.open"] = "Открыть DeepSeek",
      ["deepseek.sign_in.step_console"] = "2. На этой странице нажмите F12, откройте Console и выполните команду. Она копирует ID устройства. Если консоль запрещает вставку, сначала введите allow pasting.",
      ["deepseek.sign_in.submit"] = "Войти",
      ["deepseek.field.login"] = "Email или телефон",
      ["deepseek.field.login_placeholder"] = "you@example.com",
      ["deepseek.field.email"] = "Email",
      ["deepseek.field.mobile"] = "Телефон",
      ["deepseek.field.area_code"] = "Код страны",
      ["deepseek.field.password"] = "Пароль",
      ["deepseek.field.device_id"] = "ID устройства",
      ["deepseek.error.missing_fields"] = "Введите email или телефон, пароль и ID устройства.",
      ["deepseek.error.preparing"] = "Образ ds-web-server загружается. Повторите через минуту.",
      ["deepseek.error.container"] = "Контейнер ds-web-server не готов.",
      ["deepseek.error.sign_in_failed"] = "DeepSeek отклонил вход.",
      ["deepseek.error.invalid_credentials"] = "Неверный email, телефон или пароль.",
      ["deepseek.error.device_rejected"] = "DeepSeek отклонил ID устройства. Скопируйте его заново в браузере, где вы вошли.",
      ["deepseek.error.muted"] = "DeepSeek ограничил этот аккаунт. Повторите после окончания ограничения.",
      ["deepseek.error.banned"] = "DeepSeek заблокировал этот аккаунт.",
      ["deepseek.error.blocked"] = "DeepSeek заблокировал вход из этой сети.",
      ["deepseek.error.unavailable"] = "DeepSeek не ответил. Повторите попытку.",
    },
  },
})
