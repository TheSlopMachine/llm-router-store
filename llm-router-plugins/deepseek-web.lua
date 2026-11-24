--- @plugin DeepSeek Web
--- @author TheSlopMachine
--- @version 0.2.0
--- @plugin_api 1.8
--- @description DeepSeek web chat through ds-free-api in a Docker container: streaming, tools, files, web search
--- @license MIT
--- @allow_host chat.deepseek.com
--- @allow_host fe-static.deepseek.com
--- @allow_host hif-leim.deepseek.com
--- @require_api docker
--- @docker.image ghcr.io/niyueee/ds-free-api:v0.5.1@sha256:f906926f79b675607213cdab8f05dc339d39ebe768182fbc235d86d70ffa3074
--- @docker.memory_limit_mb 512
--- @docker.cpu_limit 1
--- @docker.container_limit 1

-- ds-free-api (github.com/NIyueeE/ds-free-api) serves the DeepSeek web chat
-- as an OpenAI API. One container holds every DeepSeek account of the
-- provider. The router owns the accounts: they are its credentials. The
-- container keeps nothing. Each start writes a fresh config without
-- accounts, and the sync job pushes the router credentials into the
-- container account pool through the ds-free-api admin API.

local docker = llm_router.docker

local IMAGE = "ghcr.io/niyueee/ds-free-api:v0.5.1@sha256:f906926f79b675607213cdab8f05dc339d39ebe768182fbc235d86d70ffa3074"
local KEY = "deepseek"
local PORT = 22217
local SEARCH_SUFFIX = "-search"
local SIGN_IN_URL = "https://chat.deepseek.com/sign_in"
local DEVICE_ID_COMMAND = "copy(SMSdk.getDeviceId())"
local DEFAULT_AREA_CODE = "+86"
local STATE_SCOPE = "container"
local ADMIN_TOKEN_TTL = 600
-- An account that failed to sign in gets one more attempt after this long.
local RETRY_SECONDS = 1800
local ACCOUNT_FIELDS = { "email", "mobile", "area_code", "password", "device_id" }

-- The client identity stays at the ds-free-api default, the Android app:
-- from a European address the CloudFront WAF challenges the desktop Chrome
-- profile at sign-in and lets the app profile through. Search runs only on
-- request.
local CORE_SETTINGS = {
  default_search_enabled = false,
}
local CORE_SETTING_NAMES = { "default_search_enabled" }

-- The router gateway address is known only inside the container, so the
-- start command writes it into the config.
local PROXY_PLACEHOLDER = "@PROXY_URL@"
local TEMPLATE_PATH = "/app/llm-router.toml"
local START_COMMAND = "mkdir -p /app/config /app/data && sed \"s#" .. PROXY_PLACEHOLDER .. "#${HTTPS_PROXY:-$HTTP_PROXY}#\" "
  .. TEMPLATE_PATH .. " > /app/config/config.toml && exec /app/ds-free-api"

-- ── Container ──

local function state_secrets()
  local stored = llm_router.storage.get(STATE_SCOPE, "secrets")
  if stored ~= nil then return stored end
  stored = { api_key = "sk-" .. llm_router.random_hex(24), admin_password = llm_router.random_hex(24) }
  llm_router.storage.set(STATE_SCOPE, "secrets", stored)
  return stored
end

local function toml_value(value)
  if type(value) == "boolean" then return tostring(value) end
  return "\"" .. value .. "\""
end

local function config_toml(api_key)
  local lines = {
    "[server]", "host = \"0.0.0.0\"", "port = " .. PORT, "",
    "[[api_keys]]", "key = \"" .. api_key .. "\"", "description = \"llm-router\"", "",
    "[ds_core]",
  }
  for _, name in ipairs(CORE_SETTING_NAMES) do
    table.insert(lines, name .. " = " .. toml_value(CORE_SETTINGS[name]))
  end
  table.insert(lines, "")
  table.insert(lines, "[proxy]")
  table.insert(lines, "url = \"" .. PROXY_PLACEHOLDER .. "\"")
  return table.concat(lines, "\n") .. "\n"
end

-- The admin API replaces the whole config. Fields left out take the
-- ds-free-api defaults. An empty account list is left out: an empty Lua
-- table encodes as an object.
local function config_document(api_key, proxy_url, accounts)
  local core = {}
  for _, name in ipairs(CORE_SETTING_NAMES) do core[name] = CORE_SETTINGS[name] end
  if #accounts > 0 then core.accounts = accounts end
  return {
    server = { host = "0.0.0.0", port = PORT },
    api_keys = { { key = api_key, description = "llm-router" } },
    ds_core = core,
    proxy = { url = proxy_url },
  }
end

local function container_spec()
  return {
    image = IMAGE,
    entrypoint = { "/bin/sh", "-c" },
    command = { START_COMMAND },
    ports = { PORT },
    files = { [TEMPLATE_PATH] = config_toml(state_secrets().api_key) },
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
-- Returns true when this call started it.
local function ensure_running(timeout_ms)
  local status, fault = docker.ensure(KEY, container_spec())
  if fault ~= nil then return false, fault end
  if status.state == "preparing" then
    return false, {
      message = "the ds-free-api image is downloading, try again in a minute",
      code = "server_error", status = 503, docker = true, reason = "preparing",
    }
  end
  local started = status.state ~= "running"
  if started then
    local _, start_fault = docker.start(KEY)
    if start_fault ~= nil then return false, start_fault end
  end
  local _, wait_fault = docker.wait(KEY, { port = PORT, timeout_ms = timeout_ms or 60000 })
  if wait_fault ~= nil then return false, wait_fault end
  return started, nil
end

local function container_request(arguments, timeout_ms)
  local client = docker.open_http(KEY, PORT, { timeout_ms = timeout_ms or 120000 })
  return client:request(arguments)
end

-- ds-free-api answers {"error": "text"} on the admin API and
-- {"error": {"message": "text"}} on /v1.
local function error_text(body)
  local ok, decoded = pcall(json.decode, body or "")
  if ok and type(decoded) == "table" then
    if type(decoded.error) == "table" and type(decoded.error.message) == "string" then return decoded.error.message end
    if type(decoded.error) == "string" then return decoded.error end
  end
  return string.sub(body or "", 1, 1000)
end

local function container_fault(operation, response)
  return {
    message = "ds-free-api " .. operation .. " failed with status " .. tostring(response.status) .. ": " .. error_text(response.body),
    code = "server_error", status = 502,
  }
end

-- ── Admin API ──

local function admin_sign_in()
  local body = json.encode({ password = state_secrets().admin_password })
  local headers = { ["content-type"] = "application/json" }
  local response, fault = container_request({ method = "POST", url = "/admin/api/login", headers = headers, body = body }, 30000)
  if fault ~= nil then return nil, fault end
  -- 403: the container started fresh and has no admin password yet.
  if response.status == 403 then
    response, fault = container_request({ method = "POST", url = "/admin/api/setup", headers = headers, body = body }, 30000)
    if fault ~= nil then return nil, fault end
  end
  if response.status == 401 then
    -- The container holds an admin password this plugin no longer knows.
    -- Remove it: the next call creates a fresh one.
    docker.remove(KEY)
    return nil, { message = "ds-free-api rejected the admin password, the container was reset", code = "server_error", status = 503 }
  end
  if response.status ~= 200 then return nil, container_fault("admin sign-in", response) end
  local token = json.decode(response.body).token
  llm_router.storage.set(STATE_SCOPE, "admin_token", token, { ttl = ADMIN_TOKEN_TTL })
  return token, nil
end

local function admin_call(method, path, payload, timeout_ms)
  for attempt = 1, 2 do
    local token = nil
    if attempt == 1 then token = llm_router.storage.get(STATE_SCOPE, "admin_token") end
    if token == nil then
      local fault
      token, fault = admin_sign_in()
      if fault ~= nil then return nil, fault end
    end
    local arguments = { method = method, url = path, headers = { authorization = "Bearer " .. token } }
    if payload ~= nil then
      arguments.headers["content-type"] = "application/json"
      arguments.body = json.encode(payload)
    end
    local response, fault = container_request(arguments, timeout_ms or 30000)
    if fault ~= nil then return nil, fault end
    if response.status ~= 401 then
      if response.status ~= 200 then return nil, container_fault(method .. " " .. path, response) end
      return json.decode(response.body), nil
    end
    -- A restarted container signs tokens with a new secret.
    llm_router.storage.delete(STATE_SCOPE, "admin_token")
  end
  return nil, { message = "ds-free-api rejected the admin token", code = "server_error", status = 502 }
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

local function same_account(left, right)
  for _, field in ipairs(ACCOUNT_FIELDS) do
    if (left[field] or "") ~= (right[field] or "") then return false end
  end
  return true
end

local function accounts_of(credential_entries)
  local accounts = {}
  for _, credential_entry in ipairs(credential_entries) do
    local account = account_of(credential_entry.data)
    if account_id(account) ~= "" then table.insert(accounts, account) end
  end
  return accounts
end

local function sorted_accounts(by_id)
  local ids = {}
  for id in pairs(by_id) do table.insert(ids, id) end
  table.sort(ids)
  local accounts = {}
  for _, id in ipairs(ids) do table.insert(accounts, by_id[id]) end
  return accounts
end

local function pool_states(status)
  local states = {}
  for _, entry in ipairs(status.accounts or {}) do
    local id = entry.email
    if id == nil or id == "" then id = entry.mobile end
    if id ~= nil and id ~= "" then states[id] = entry.state end
  end
  return states
end

local function put_accounts(config, by_id)
  local document = config_document(state_secrets().api_key, (config.proxy or {}).url, sorted_accounts(by_id))
  -- The call returns after every new account signed in.
  local _, fault = admin_call("PUT", "/admin/api/config", document, 900000)
  return fault
end

-- Makes the container account pool hold the given accounts.
--   merge: keep accounts the list does not name.
--   drop: ids to remove.
--   retry_missing: "always" or "cooldown". A configured account that is
--     not in the pool failed to sign in. ds-free-api signs it in again
--     on the next config write.
-- Returns the pool state per account id.
local function sync_accounts(accounts, options)
  options = options or {}
  local config, fault = admin_call("GET", "/admin/api/config")
  if fault ~= nil then return nil, fault end
  local status, status_fault = admin_call("GET", "/admin/api/status")
  if status_fault ~= nil then return nil, status_fault end
  local states = pool_states(status)

  local current = {}
  for _, account in ipairs((config.ds_core or {}).accounts or {}) do current[account_id(account)] = account_of(account) end
  local target = {}
  if options.merge then
    for id, account in pairs(current) do target[id] = account end
  end
  for _, account in ipairs(accounts) do target[account_id(account)] = account end
  for id in pairs(options.drop or {}) do target[id] = nil end

  local changed, missing, stale = false, false, {}
  for id, account in pairs(target) do
    local existing = current[id]
    if existing == nil then
      changed = true
    elseif not same_account(existing, account) then
      -- ds-free-api keeps a pooled account as it signed in: drop it first.
      changed = true
      stale[id] = true
    elseif states[id] == nil then
      missing = true
    end
  end
  for id in pairs(current) do
    if target[id] == nil then changed = true end
  end

  if not changed then
    if not missing or options.retry_missing == nil then return states, nil end
    if options.retry_missing == "cooldown" then
      if llm_router.storage.get(STATE_SCOPE, "retry") ~= nil then return states, nil end
      llm_router.storage.set(STATE_SCOPE, "retry", true, { ttl = RETRY_SECONDS })
    end
  end

  if next(stale) ~= nil then
    local kept = {}
    for id, account in pairs(target) do
      if not stale[id] then kept[id] = account end
    end
    local put_fault = put_accounts(config, kept)
    if put_fault ~= nil then return nil, put_fault end
  end
  local put_fault = put_accounts(config, target)
  if put_fault ~= nil then return nil, put_fault end
  llm_router.log("info", "account pool synchronized", { count = #sorted_accounts(target), operation = options.merge and "merge" or "replace" })

  local after, after_fault = admin_call("GET", "/admin/api/status")
  if after_fault ~= nil then return nil, after_fault end
  return pool_states(after), nil
end

-- The newest sign-in failure ds-free-api logged for one account.
local function sign_in_failure(id)
  local page = admin_call("GET", "/admin/api/runtime-logs?offset=0&limit=200")
  if page == nil then return nil end
  local marker = "during sync " .. id .. ": "
  for _, entry in ipairs(page.logs or {}) do
    local message = entry.message or ""
    local start = string.find(message, marker, 1, true)
    if start ~= nil then return string.sub(message, start + #marker) end
  end
  return nil
end

-- Runs one call against the container. A stopped container starts, takes
-- the accounts of this request, and the call runs again.
local function with_container(run)
  local result, fault = run()
  if not stopped(fault) then return result, fault end
  local _, start_fault = ensure_running()
  if start_fault ~= nil then return nil, start_fault end
  local _, sync_fault = sync_accounts(accounts_of(llm_router.credentials.list()), { merge = true })
  if sync_fault ~= nil then return nil, sync_fault end
  return run()
end

-- ── Chat ──

local function api_headers()
  return { ["content-type"] = "application/json", authorization = "Bearer " .. state_secrets().api_key }
end

-- "<model>-search" is the same model with DeepSeek web search on.
local function upstream_model(model_name)
  if #model_name > #SEARCH_SUFFIX and string.sub(model_name, -#SEARCH_SUFFIX) == SEARCH_SUFFIX then
    return string.sub(model_name, 1, #model_name - #SEARCH_SUFFIX), true
  end
  return model_name, false
end

-- Thinking stays off. ds-free-api v0.5.1 reads the current DeepSeek
-- thinking stream wrong: the answer joins the reasoning and the content
-- comes back empty (github.com/NIyueeE/ds-free-api/issues/125).
local THINKING_EFFORT = "none"

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
    reasoning_effort = THINKING_EFFORT,
    stream = stream,
  }
  if stream then payload.stream_options = { include_usage = true } end
  if search then payload.web_search_options = { search_context_size = "high" } end
  return payload
end

local function upstream_error(status, body, retry_after)
  local message = "ds-free-api status " .. tostring(status) .. ": " .. error_text(body)
  if status == 429 then
    local result = { message = message, code = "rate_limit", status = 429 }
    if retry_after ~= nil then result.headers = { ["Retry-After"] = retry_after } end
    return result
  end
  if status == 400 then
    -- ds-free-api names an unknown model with this text.
    if string.find(message, "不支持的模型", 1, true) ~= nil then
      return { message = message, code = "not_found", status = 404 }
    end
    return { message = message, code = "invalid_request_error", status = 400 }
  end
  return { message = message, code = "upstream_error", status = 502 }
end

local function post_chat(payload, timeout_ms)
  local response, fault = with_container(function()
    return container_request({
      method = "POST", url = "/v1/chat/completions", headers = api_headers(), body = json.encode(payload),
    }, timeout_ms or 900000)
  end)
  if fault ~= nil then return nil, fault end
  if response.status ~= 200 then
    return nil, upstream_error(response.status, response.body, response.headers["retry-after"])
  end
  return json.decode(response.body), nil
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
    capabilities = { "tools" },
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
    local _, fault = ensure_running(20000)
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
        local _, fault = ensure_running(120000)
        if fault ~= nil then
          if fault.reason == "preparing" then return true end
          return fault
        end
        local _, sync_fault = sync_accounts(accounts_of(llm_router.credentials.list()), { retry_missing = "cooldown" })
        return sync_fault
      end,
    },
  },

  auth_initiate = function(ctx) return sign_in_page() end,

  auth_step = function(ctx, input)
    local values = input.values or {}
    if input.action ~= "sign_in" then return sign_in_page(values) end
    local account, problem = account_from_form(values)
    if problem ~= nil then return sign_in_page(values, problem) end

    local _, fault = ensure_running(90000)
    if fault ~= nil then
      if fault.reason == "preparing" then return sign_in_page(values, "deepseek.error.preparing") end
      return sign_in_page(values, "deepseek.error.container", fault.message)
    end
    local id = account_id(account)
    local states, sync_fault = sync_accounts({ account }, { merge = true, retry_missing = "always" })
    if sync_fault ~= nil then return sign_in_page(values, "deepseek.error.container", sync_fault.message) end
    local state = states[id]
    if state == "idle" or state == "busy" then
      llm_router.log("info", "account signed in", { operation = "auth_step" })
      return { credentials = account }
    end

    local reason = sign_in_failure(id) or state or "no detail"
    -- A failed account leaves the pool so it does not sign in again with
    -- the same data.
    local _, drop_fault = sync_accounts({}, { merge = true, drop = { [id] = true } })
    if drop_fault ~= nil then
      llm_router.log("warn", "failed account stays in the pool", { error = drop_fault.message })
    end
    llm_router.log("warn", "account sign-in failed", { reason = reason, operation = "auth_step" })
    return sign_in_page(values, "deepseek.error.sign_in_failed", reason)
  end,

  check_health = function(ctx, credential)
    local account = account_of(credential.data)
    local id = account_id(account)
    local _, fault = ensure_running(60000)
    if fault ~= nil then return { status = "unknown", message = fault.message } end
    local config, config_fault = admin_call("GET", "/admin/api/config")
    if config_fault ~= nil then return { status = "unknown", message = config_fault.message } end
    local configured = false
    for _, entry in ipairs((config.ds_core or {}).accounts or {}) do
      if account_id(account_of(entry)) == id and same_account(account_of(entry), account) then configured = true end
    end
    if not configured then return { status = "unknown", message = "the account is not in the container yet, the sync job adds it within a minute" } end
    local status, status_fault = admin_call("GET", "/admin/api/status")
    if status_fault ~= nil then return { status = "unknown", message = status_fault.message } end
    local state = pool_states(status)[id]
    if state == "idle" or state == "busy" then return { status = "healthy" } end
    if state == "error" then return { status = "unknown", message = "sign-in is being retried" } end
    if state == "invalid" then return { status = "unhealthy", message = "DeepSeek muted or banned the account" } end
    return { status = "unhealthy", message = "sign-in failed: " .. (sign_in_failure(id) or "no detail") }
  end,

  complete = function(ctx, request)
    return post_chat(chat_payload(request, false))
  end,

  complete_stream = function(ctx, request, emit)
    local client = docker.open_http(KEY, PORT, { timeout_ms = 900000 })
    local arguments = {
      method = "POST", url = "/v1/chat/completions", headers = api_headers(),
      body = json.encode(chat_payload(request, true)),
      on_response = function(head)
        if head.status ~= 200 then return upstream_error(head.status, head.body, head.headers["retry-after"]) end
      end,
      on_line = function(line)
        local payload = string.match(line, "^data: (.*)$")
        if payload == nil or payload == "[DONE]" then return end
        local chunk = json.decode(payload)
        if (chunk.choices ~= nil and #chunk.choices > 0) or chunk.usage ~= nil then emit(chunk) end
      end,
    }
    local _, fault = with_container(function() return client:stream(arguments) end)
    return nil, fault
  end,

  get_model_infos = function(ctx)
    local response, fault = with_container(function()
      return container_request({ method = "GET", url = "/v1/models", headers = api_headers() }, 30000)
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
    local result, fault = post_chat(chat_payload({
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
      ["deepseek.error.preparing"] = "The ds-free-api image is downloading. Try again in a minute.",
      ["deepseek.error.container"] = "The ds-free-api container is not ready.",
      ["deepseek.error.sign_in_failed"] = "DeepSeek rejected the sign-in.",
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
      ["deepseek.error.preparing"] = "Образ ds-free-api загружается. Повторите через минуту.",
      ["deepseek.error.container"] = "Контейнер ds-free-api не готов.",
      ["deepseek.error.sign_in_failed"] = "DeepSeek отклонил вход.",
    },
  },
})
