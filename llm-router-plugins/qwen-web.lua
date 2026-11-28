--- @plugin Qwen Web
--- @author TheSlopMachine
--- @version 0.3.3
--- @plugin_api 1.9
--- @description Qwen web chat (chat.qwen.ai) with your account through a headless browser in Docker: streaming, thinking, web search
--- @license MIT
--- @allow_host chat.qwen.ai
--- @allow_host auth.qwen.ai
--- @allow_host assets.alicdn.com
--- @allow_host g.alicdn.com
--- @allow_host at.alicdn.com
--- @allow_host img.alicdn.com
--- @allow_host o.alicdn.com
--- @allow_host d.alicdn.com
--- @allow_host alidt.alicdn.com
--- @allow_host sg-wum.alibaba.com
--- @allow_host fourier.alibaba.com
--- @allow_host fourier.taobao.com
--- @allow_host aplus.qwen.ai
--- @allow_host ss.qwen.ai
--- @allow_host dw-data-channel-sg.ucweb.com
--- @require_api docker
--- @docker.image ghcr.io/theslopmachine/qwen-web-egress:v1.0.0@sha256:895e6c58a0e18a33ae64c110b01aad3c05038966f35ac156ef75d12d1a4a9fe0
--- @docker.memory_limit_mb 1536
--- @docker.cpu_limit 2
--- @docker.pids_limit 2048
--- @docker.container_limit 1

-- chat.qwen.ai serves the web chat through a JSON and SSE API. A credential
-- is an account: the email and the SHA-256 of its password, the form the
-- sign-in endpoint takes. Access tokens live 15 minutes and refresh with a
-- refresh token; both stay in storage per account, never in the credential.
-- Every request opens a fresh chat, sends the whole conversation as one user
-- turn and deletes the chat afterwards.
--
-- Qwen risk control blocks chat calls that do not come from its page. The
-- qwen-web-egress container (github.com/TheSlopMachine/qwen-web-egress)
-- keeps one headless Chromium page per account on chat.qwen.ai and runs every chat.qwen.ai call
-- as a fetch inside it: the page's anti-bot script signs the call. Sign-in
-- and token refresh go to auth.qwen.ai directly. The extra hosts are the
-- page's scripts, fonts and telemetry; its anti-bot script needs them.

local docker = llm_router.docker

local AUTH_BASE = "https://auth.qwen.ai"
local CHAT_ORIGIN = "https://chat.qwen.ai"
local SIGN_IN_URL = CHAT_ORIGIN .. "/auth"
local IMAGE = "ghcr.io/theslopmachine/qwen-web-egress:v1.0.0@sha256:895e6c58a0e18a33ae64c110b01aad3c05038966f35ac156ef75d12d1a4a9fe0"
local KEY = "qwen"
local PORT = 8790
-- The web front end version and bx-v the page sends. Sign-in sends the
-- static bx-ua the page sends when its anti-bot script fails to load: the
-- auth API accepts it.
local WEB_VERSION = "0.3.26"
local BX_VERSION = "2.5.37"
local USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
local SEARCH_SUFFIX = "-search"
local SESSION_SCOPE = "session"
-- A token this close to its expiry is renewed before use.
local TOKEN_MARGIN_SECONDS = 60
local REQUEST_TIMEOUT_MS = 30000
-- The first call of an account opens its page: up to two minutes.
local EGRESS_TIMEOUT_MS = 180000
local STREAM_TIMEOUT_MS = 900000
-- Parking times per outcome. Risk control blocks a burst from one account;
-- quota resets on a daily window that the API does not state.
local RISK_PARK_SECONDS = 60
local MAX_RISK_PARK_SECONDS = 3600
local RISK_STREAK_SCOPE = "risk_streak"
-- account/model pairs that reached the model's daily limit.
local MODEL_LIMIT_SCOPE = "model_limit"
-- Accounts whose memory was checked off recently.
local MEMORY_SCOPE = "memory_off"
local MEMORY_CHECK_SECONDS = 24 * 3600
-- A streak older than this is forgotten.
local RISK_STREAK_SECONDS = 2 * 3600
local DEFAULT_PARK_SECONDS = 60
local QUOTA_PARK_SECONDS = 3600
-- A dead account waits this long when the provider does not allow disabling
-- credentials.
local DEAD_PARK_SECONDS = 3600
local MAX_PARK_SECONDS = 90 * 24 * 3600
-- Raised from on_line to close a stream that has delivered its answer: the
-- server may keep the connection open after the final frame.
local STREAM_STOP = "qwen-web: stream complete"

-- ── Helpers ──

local function trim(value)
  if type(value) ~= "string" then return "" end
  return (string.gsub(value, "^%s*(.-)%s*$", "%1"))
end

local function excerpt(text, size)
  return string.sub(text or "", 1, size or 500)
end

local function contains(text, needle)
  return string.find(text, needle, 1, true) ~= nil
end

local function decode_json(text)
  local ok, decoded = pcall(json.decode, text or "")
  if ok and type(decoded) == "table" then return decoded end
  return nil
end

local function uuid4()
  local hex = llm_router.random_hex(16)
  local variant_index = tonumber(string.sub(hex, 17, 17), 16) % 4 + 1
  return string.sub(hex, 1, 8) .. "-" .. string.sub(hex, 9, 12) .. "-4" .. string.sub(hex, 14, 16)
    .. "-" .. string.sub("89ab", variant_index, variant_index) .. string.sub(hex, 18, 20)
    .. "-" .. string.sub(hex, 21, 32)
end

local function jwt_claims(token)
  local middle = string.match(token or "", "^[^%.]+%.([^%.]+)%.")
  if middle == nil then return nil end
  middle = string.gsub(string.gsub(middle, "-", "+"), "_", "/")
  middle = middle .. string.rep("=", (4 - #middle % 4) % 4)
  local ok, decoded = pcall(function() return json.decode(llm_router.base64_decode(middle)) end)
  if ok and type(decoded) == "table" then return decoded end
  return nil
end

local function timezone_header()
  return os.date("!%a %b %d %Y %H:%M:%S") .. " GMT+0000"
end

-- The header set of the web page for its same-site calls to auth.qwen.ai.
local function auth_headers()
  local bx = "defaultFY2_load_failed with timeout@@" .. CHAT_ORIGIN .. "@@" .. tostring(os.time() * 1000)
  return {
    ["accept"] = "application/json, text/plain, */*",
    ["accept-language"] = "en-US,en;q=0.9",
    ["content-type"] = "application/json",
    ["user-agent"] = USER_AGENT,
    ["origin"] = CHAT_ORIGIN,
    ["referer"] = CHAT_ORIGIN .. "/",
    ["source"] = "web",
    ["version"] = WEB_VERSION,
    ["timezone"] = timezone_header(),
    ["x-request-id"] = uuid4(),
    ["x-accel-buffering"] = "no",
    ["bx-v"] = BX_VERSION,
    ["bx-ua"] = bx,
    ["bx-umidtoken"] = bx,
    ["sec-fetch-dest"] = "empty",
    ["sec-fetch-mode"] = "cors",
    ["sec-fetch-site"] = "same-site",
    ["sec-ch-ua"] = "\"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"156\", \"Google Chrome\";v=\"156\"",
    ["sec-ch-ua-mobile"] = "?0",
    ["sec-ch-ua-platform"] = "\"macOS\"",
    ["x-request-origin"] = CHAT_ORIGIN,
  }
end

-- The headers the page's own front end sets on its API calls. The browser
-- adds the rest, and the anti-bot script adds bx-ua, bx-umidtoken and bx-v.
local function page_headers(token)
  return {
    ["accept"] = "application/json",
    ["content-type"] = "application/json",
    ["authorization"] = "Bearer " .. token,
    ["source"] = "web",
    ["version"] = WEB_VERSION,
    ["timezone"] = timezone_header(),
    ["x-request-id"] = uuid4(),
    ["x-accel-buffering"] = "no",
  }
end

local function client(timeout_ms)
  return llm_router.http_client({ timeout_ms = timeout_ms or REQUEST_TIMEOUT_MS })
end

-- ── Outcomes ──

-- A problem is {kind, message, wait?, fault?}. Kinds:
--   risk, challenge, rate, quota: transient, the account parks.
--   model_limit: the daily limit of one model, the account skips that model.
--   invalid_credentials, auth: Qwen rejects the account.
--   blocked: sign-in blocked by risk control, the account parks.
--   bad_request, not_found: the request itself is wrong.
--   server, transport, unavailable: Qwen failed; no account is blamed.
local function problem(kind, message, extra)
  local result = extra or {}
  result.kind = kind
  result.message = message
  return result
end

local function is_risk(text)
  return contains(text, "RGV587") or contains(text, "FAIL_SYS_USER_VALIDATE") or contains(text, "aliyun_waf")
end

local QUOTA_WORDS = { "quota", "credits", "insufficient", "not enough", "额度", "积分", "余额" }

-- Maps an answer that is not a stream to a problem. Qwen wraps errors as
-- {success=false, data={code, details}}, often with HTTP 200; error frames
-- carry {code, details, message}.
local function classify(status, text)
  text = text or ""
  local summary = "Qwen answered " .. tostring(status) .. ": " .. excerpt(text, 300)
  if is_risk(text) then return problem("risk", summary) end
  local decoded = decode_json(text) or {}
  local fields = type(decoded.data) == "table" and decoded.data or decoded
  local code = type(fields.code) == "string" and fields.code or ""
  local details = string.lower((type(fields.details) == "string" and fields.details or "")
    .. " " .. (type(fields.message) == "string" and fields.message or ""))
  if status == 401 or status == 403 or code == "Unauthorized"
    or contains(details, "token has expired") or contains(details, "please log in") then
    return problem("auth", summary)
  end
  if status == 429 or code == "RateLimited" then return problem("rate", summary) end
  for _, word in ipairs(QUOTA_WORDS) do
    if contains(details, word) or contains(string.lower(code), word) then return problem("quota", summary) end
  end
  if status == 404 or contains(details, "model not found") then return problem("not_found", summary) end
  if status == 400 or code == "Bad_Request" or code == "RequestValidationError" then
    return problem("bad_request", summary)
  end
  return problem("server", summary)
end

local function call_error(issue)
  local kind = issue.kind
  if kind == "transport" and issue.fault ~= nil then return issue.fault end
  if kind == "risk" or kind == "challenge" or kind == "rate" or kind == "quota" or kind == "model_limit" or kind == "blocked" then
    return { message = issue.message, code = "rate_limit", status = 429 }
  end
  if kind == "auth" or kind == "invalid_credentials" then
    return { message = issue.message, code = "auth_error", status = 401 }
  end
  if kind == "bad_request" then return { message = issue.message, code = "invalid_request_error", status = 400 } end
  if kind == "not_found" then return { message = issue.message, code = "not_found", status = 404 } end
  return { message = issue.message, code = "upstream_error", status = 502 }
end

local function park(credential_entry, seconds, reason, detail)
  seconds = math.min(math.max(seconds, 1), MAX_PARK_SECONDS)
  pcall(llm_router.credentials.park, credential_entry.id, seconds, reason)
  llm_router.log("warn", "credential parked", { cred_id = credential_entry.id, wait = seconds, reason = reason, error = detail })
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

-- ── Container ──

local function container_spec()
  return {
    image = IMAGE,
    ports = { PORT },
    restart = "on_failure",
    -- Node's fetch ignores the gateway proxy variables, so the probe stays local.
    health = {
      command = { "node", "-e", "fetch('http://127.0.0.1:" .. PORT .. "/health').then((r) => process.exit(r.ok ? 0 : 1), () => process.exit(1))" },
      interval_seconds = 5,
      timeout_seconds = 5,
      start_period_seconds = 10,
      retries = 12,
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
      message = "the Qwen browser image is being prepared, try again in a few minutes",
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

-- Runs one call against the container and starts a stopped container once.
local function with_container(run)
  local result, fault = run()
  if not stopped(fault) then return result, fault end
  local start_fault = ensure_running()
  if start_fault ~= nil then return nil, start_fault end
  return run()
end

-- An answer of the container itself, not of chat.qwen.ai, carries
-- x-egress-error. A captcha or a stuck page is a challenge: the container
-- drops that page, and a fresh one usually passes.
local function egress_issue(headers, body)
  local code = (headers or {})["x-egress-error"]
  if code == nil then return nil end
  if code == "risk_challenge" or code == "timeout" then
    return problem("challenge", "Qwen held the browser request (" .. code .. ")")
  end
  local decoded = decode_json(body)
  local message = decoded and type(decoded.error) == "table" and decoded.error.message or excerpt(body, 300)
  return problem("server", "Qwen browser " .. code .. ": " .. tostring(message))
end

-- detach makes the container answer 202 at once and run the call in the
-- background: for calls whose answer nobody waits for.
local function egress_body(account, token, method, path, body, detach)
  return json.encode({
    account = account.email, method = method, path = path, headers = page_headers(token), body = body, detach = detach,
  })
end

-- Runs one chat.qwen.ai API call as a fetch in the account's page.
local function page_request(account, token, method, path, body, detach)
  local container_client = docker.open_http(KEY, PORT, { timeout_ms = EGRESS_TIMEOUT_MS })
  local response, fault = with_container(function()
    return container_client:request({
      method = "POST", url = "/fetch", headers = { ["content-type"] = "application/json" },
      body = egress_body(account, token, method, path, body, detach),
    })
  end)
  if fault ~= nil then return nil, problem("transport", fault.message, { fault = fault }) end
  local issue = egress_issue(response.headers, response.body)
  if issue ~= nil then return nil, issue end
  return response, nil
end

-- ── Sessions ──

local function session_key(email)
  return string.lower(email)
end

-- Reads a sign-in or refresh answer: {data={access_token, refresh_token}}.
local function session_from(email, response, previous)
  local decoded = decode_json(response.body)
  local data = decoded and decoded.data
  if response.status ~= 200 or type(data) ~= "table" or type(data.access_token) ~= "string" or data.access_token == "" then
    if is_risk(response.body) then return nil, problem("blocked", "Qwen risk control blocked the sign-in") end
    local code = type(data) == "table" and data.code or nil
    if code == "Bad_Request" or code == "Unauthorized" then
      return nil, problem("invalid_credentials", type(data.details) == "string" and data.details or code)
    end
    return nil, problem("unavailable", "Qwen answered " .. tostring(response.status) .. ": " .. excerpt(response.body, 300))
  end
  local claims = jwt_claims(data.access_token)
  if claims == nil or type(claims.exp) ~= "number" then
    return nil, problem("unavailable", "the Qwen access token carries no expiry")
  end
  local session = {
    access_token = data.access_token,
    refresh_token = (type(data.refresh_token) == "string" and data.refresh_token ~= "") and data.refresh_token
      or (previous and previous.refresh_token),
    expires_at = claims.exp,
  }
  llm_router.storage.set(SESSION_SCOPE, session_key(email), session)
  return session, nil
end

local function sign_in(email, password_hash)
  local response, fault = client():request({
    method = "POST", url = AUTH_BASE .. "/api/v2/auths/signin",
    headers = auth_headers(),
    body = json.encode({ email = email, password = password_hash }),
  })
  if fault ~= nil then return nil, problem("unavailable", fault.message) end
  return session_from(email, response, nil)
end

local function refresh(email, previous)
  local headers = auth_headers()
  headers["cookie"] = "refresh_token=" .. previous.refresh_token
  local response, fault = client():request({
    method = "GET", url = AUTH_BASE .. "/api/v2/auths/refresh", headers = headers,
  })
  if fault ~= nil then return nil, problem("unavailable", fault.message) end
  return session_from(email, response, previous)
end

-- Returns a usable access token. renew skips the stored token: the caller
-- saw it rejected. A failed refresh falls back to a full sign-in only when
-- Qwen rejected the refresh token itself.
local function access_token(account, renew)
  local session = llm_router.storage.get(SESSION_SCOPE, session_key(account.email))
  if not renew and session ~= nil and session.expires_at - os.time() > TOKEN_MARGIN_SECONDS then
    return session.access_token, nil
  end
  if session ~= nil and session.refresh_token ~= nil then
    local refreshed, issue = refresh(account.email, session)
    if refreshed ~= nil then return refreshed.access_token, nil end
    if issue.kind ~= "invalid_credentials" then return nil, issue end
  end
  local signed_in, issue = sign_in(account.email, account.password_hash)
  if signed_in == nil then return nil, issue end
  llm_router.log("info", "account signed in", { operation = "sign_in" })
  return signed_in.access_token, nil
end

-- ── Account memory ──

-- Qwen memory carries facts from one chat into the next, so one router
-- request would leak into another. The plugin turns it off for each account
-- and checks again once a day: the user may turn it back on. A failure only
-- logs; the request goes on.
local function disable_memory(account, token)
  local key = session_key(account.email)
  if llm_router.storage.get(MEMORY_SCOPE, key) ~= nil then return end
  local response, issue = page_request(account, token, "GET", "/api/v2/users/user/settings", nil)
  local decoded = response and decode_json(response.body)
  local settings = decoded and decoded.data
  if issue ~= nil or response.status ~= 200 or type(settings) ~= "table" then
    llm_router.log("warn", "account memory not checked", {
      operation = "memory", error = issue and issue.message or excerpt(response and response.body, 300),
    })
    return
  end
  local memory = type(settings.memory) == "table" and settings.memory or {}
  if memory.enable_memory ~= false or memory.enable_history_memory ~= false then
    -- Only the memory section goes up: the endpoint merges it into the rest.
    local update, update_issue = page_request(account, token, "POST", "/api/v2/users/user/settings/update", json.encode({
      memory = {
        enable_memory = false,
        enable_history_memory = false,
        memory_version_reminder = memory.memory_version_reminder == true,
      },
    }))
    local answer = update and decode_json(update.body)
    if update_issue ~= nil or update.status ~= 200 or (answer and answer.success == false) then
      llm_router.log("warn", "account memory not turned off", {
        operation = "memory", error = update_issue and update_issue.message or excerpt(update and update.body, 300),
      })
      return
    end
    llm_router.log("info", "account memory turned off", { operation = "memory" })
  end
  llm_router.storage.set(MEMORY_SCOPE, key, true, { ttl = MEMORY_CHECK_SECONDS })
end

-- ── Chat ──

local function text_of(content)
  if type(content) == "string" then return content end
  if type(content) ~= "table" then return "" end
  local pieces = {}
  for _, part in ipairs(content) do
    if part.type == "text" and type(part.text) == "string" then table.insert(pieces, part.text) end
  end
  return table.concat(pieces, "\n")
end

local ROLE_LABELS = { system = "System", developer = "System", user = "User", assistant = "Assistant", tool = "Tool result" }

-- Qwen takes one user turn per request. A single user message goes as is;
-- a conversation becomes a labelled transcript.
local function prompt_of(messages)
  if #messages == 1 and messages[1].role == "user" then return text_of(messages[1].content) end
  local blocks = {}
  for _, message in ipairs(messages) do
    local text = text_of(message.content)
    for _, tool_call in ipairs(message.tool_calls or {}) do
      local called = tool_call["function"] or {}
      text = text .. "\nTool call: " .. tostring(called.name) .. "(" .. tostring(called.arguments) .. ")"
    end
    table.insert(blocks, (ROLE_LABELS[message.role] or "User") .. ":\n" .. trim(text))
  end
  return table.concat(blocks, "\n\n")
end

-- "<model>-search" is the same model with Qwen web search on.
local function upstream_model(model_name)
  if #model_name > #SEARCH_SUFFIX and string.sub(model_name, -#SEARCH_SUFFIX) == SEARCH_SUFFIX then
    return string.sub(model_name, 1, #model_name - #SEARCH_SUFFIX), true
  end
  return model_name, false
end

-- Qwen modes: Fast (no thinking), Auto (the model decides), Thinking.
local function feature_config(effort, search)
  local config = { output_schema = "phase", research_mode = "normal", auto_search = search }
  if effort == "none" or effort == "minimal" then
    config.thinking_enabled = false
    config.auto_thinking = false
    config.thinking_mode = "Fast"
    return config
  end
  config.thinking_enabled = true
  config.thinking_format = "summary"
  if effort == "high" or effort == "xhigh" then
    config.auto_thinking = false
    config.thinking_mode = "Thinking"
  else
    config.auto_thinking = true
    config.thinking_mode = "Auto"
  end
  return config
end

local function completion_body(chat_id, model, search, request)
  local now = os.time()
  return json.encode({
    stream = true, version = "2.1", incremental_output = true,
    chatId = chat_id, chat_id = chat_id, chat_mode = "normal", model = model,
    parentId = "", parent_id = json.null, timestamp = now,
    messages = { {
      id = json.null, fid = uuid4(), parentId = json.null, parent_id = json.null, childrenIds = { uuid4() },
      role = "user", content = prompt_of(request.messages), user_action = "chat", files = json.array({}),
      timestamp = now, models = { model }, model = "", chat_type = "t2t", sub_chat_type = "t2t",
      feature_config = feature_config(request.reasoning_effort, search),
      extra = { meta = { subChatType = "t2t" } },
    } },
  })
end

local function create_chat(account, token, model)
  local response, issue = page_request(account, token, "POST", "/api/v2/chats/new", json.encode({
    title = "New Chat", models = { model }, chat_mode = "normal", chat_type = "t2t",
    timestamp = os.time() * 1000, project_id = "",
  }))
  if issue ~= nil then return nil, issue end
  local decoded = decode_json(response.body)
  local chat_id = decoded and decoded.success ~= false and type(decoded.data) == "table" and decoded.data.id or nil
  if response.status ~= 200 or type(chat_id) ~= "string" then return nil, classify(response.status, response.body) end
  return chat_id, nil
end

-- Deletes the chat so the account history stays clean. The container runs
-- the call in the background and logs its failure: the answer already
-- arrived.
local function delete_chat(account, token, chat_id)
  local response, issue = page_request(account, token, "DELETE", "/api/v2/chats/" .. chat_id, nil, true)
  if issue ~= nil or response.status ~= 202 then
    llm_router.log("warn", "chat not deleted", {
      operation = "delete_chat", status = response and response.status or nil,
      error = issue and issue.message or excerpt(response.body, 200),
    })
  end
end

-- thinking_summary frames repeat the whole summary so far: titles and
-- thoughts are arrays that grow. Only the new text goes out.
local function emit_summary(state, extra, sink)
  local titles = ((extra or {}).summary_title or {}).content or {}
  local thoughts = ((extra or {}).summary_thought or {}).content or {}
  for i = 1, math.max(#titles, #thoughts) do
    local title = titles[i]
    if type(title) == "string" and state.titles[i] == nil then
      state.titles[i] = true
      sink("reasoning", (i > 1 and "\n" or "") .. title .. "\n")
    end
    local thought = thoughts[i]
    local sent = state.thoughts[i] or 0
    if type(thought) == "string" and #thought > sent then
      state.thoughts[i] = #thought
      sink("reasoning", string.sub(thought, sent + 1) .. (sent == 0 and "\n" or ""))
    end
  end
end

-- One stream can carry two generations (two response.created frames). The
-- first response_id that sends a choice is followed, the other is dropped.
local function handle_line(state, line, sink)
  local data = string.match(line, "^data:%s?(.*)$")
  if data == nil then
    if line ~= "" and #state.plain < 64 then table.insert(state.plain, line) end
    return
  end
  if data == "" or data == "[DONE]" then return end
  local frame = decode_json(data)
  if frame == nil then return end
  if frame.error ~= nil then
    state.issue = classify(200, type(frame.error) == "table" and json.encode(frame.error) or tostring(frame.error))
    error(STREAM_STOP)
  end
  local choice = type(frame.choices) == "table" and frame.choices[1] or nil
  local delta = type(choice) == "table" and choice.delta or nil
  if type(delta) ~= "table" then return end
  if frame.response_id ~= nil then
    if state.response_id == nil then state.response_id = frame.response_id end
    if frame.response_id ~= state.response_id then return end
  end
  if type(frame.usage) == "table" then state.usage = frame.usage end
  local phase = delta.phase
  if phase == "answer" then
    if type(delta.extra) == "table" and delta.extra.update_member == true then
      -- The answer text names the limit, for example "You've reached
      -- today's Qwen4-Max limit": it is per model, not per account.
      local text = type(delta.content) == "string" and delta.content or ""
      state.issue = problem("model_limit", text ~= "" and text or "Qwen reached the limit of this model")
      error(STREAM_STOP)
    end
    if type(delta.content) == "string" and delta.content ~= "" then sink("content", delta.content) end
    if delta.status == "finished" then
      state.finished = true
      error(STREAM_STOP)
    end
  elseif phase == "thinking_summary" then
    emit_summary(state, delta.extra, sink)
  elseif phase == "think" then
    if type(delta.content) == "string" and delta.content ~= "" then sink("reasoning", delta.content) end
  end
end

-- Runs one completion. sink(kind, text) receives "reasoning" and "content"
-- in order. Returns the upstream usage or a problem.
local function run_completion(account, token, model, search, request, sink)
  local started_at = os.time()
  local chat_id, chat_issue = create_chat(account, token, model)
  if chat_issue ~= nil then
    llm_router.log("warn", "chat not created", { operation = "create_chat", reason = chat_issue.kind, error = chat_issue.message })
    return nil, chat_issue
  end
  local chat_created_at = os.time()
  local state = { titles = {}, thoughts = {}, plain = {} }
  local head_issue = nil
  local body = egress_body(account, token, "POST", "/api/v2/chat/completions?chat_id=" .. chat_id,
    completion_body(chat_id, model, search, request))
  local container_client = docker.open_http(KEY, PORT, { timeout_ms = STREAM_TIMEOUT_MS })
  local _, fault = with_container(function()
    return container_client:stream({
      method = "POST", url = "/fetch", headers = { ["content-type"] = "application/json" }, body = body,
      on_response = function(head)
        if head.status ~= 200 then
          head_issue = egress_issue(head.headers, head.body) or classify(head.status, head.body)
          return { message = head_issue.message, code = "upstream_error", status = 502 }
        end
      end,
      on_line = function(line) handle_line(state, line, sink) end,
    })
  end)
  local answered_at = os.time()
  delete_chat(account, token, chat_id)
  llm_router.log("debug", "completion timings", {
    operation = "completion", model = model,
    create_seconds = chat_created_at - started_at,
    answer_seconds = answered_at - chat_created_at,
    delete_seconds = os.time() - answered_at,
    finished = state.finished == true,
  })
  if head_issue ~= nil then return nil, head_issue end
  if state.finished then return state.usage, nil end
  if state.issue ~= nil then return nil, state.issue end
  if fault ~= nil then return nil, problem("transport", fault.message, { fault = fault }) end
  if #state.plain > 0 then return nil, classify(200, table.concat(state.plain, "\n")) end
  return nil, problem("server", "the Qwen stream ended before the answer finished")
end

local function account_of(data)
  data = data or {}
  if type(data.email) ~= "string" or data.email == "" or type(data.password_hash) ~= "string" then return nil end
  return { email = data.email, password_hash = data.password_hash }
end

-- One account: a token, the completion, and one more try when the first
-- fails before any output. A rejected token is renewed for the second try;
-- a challenged page is replaced by the container. A second failure of the
-- same kind blames the account.
local function attempt(account, model, search, request, output)
  local token, issue = access_token(account, false)
  if token == nil then return nil, issue end
  disable_memory(account, token)
  local usage, completion_issue = run_completion(account, token, model, search, request, output.sink)
  if completion_issue == nil or output.delivered()
    or (completion_issue.kind ~= "auth" and completion_issue.kind ~= "challenge") then
    return usage, completion_issue
  end
  if completion_issue.kind == "auth" then
    token, issue = access_token(account, true)
    if token == nil then return nil, issue end
  end
  local first_kind = completion_issue.kind
  output.reset()
  usage, completion_issue = run_completion(account, token, model, search, request, output.sink)
  if completion_issue ~= nil and completion_issue.kind == first_kind then
    completion_issue.kind = first_kind == "auth" and "invalid_credentials" or "risk"
  end
  return usage, completion_issue
end

local PARK_SECONDS = {
  risk = RISK_PARK_SECONDS, challenge = RISK_PARK_SECONDS, blocked = RISK_PARK_SECONDS,
  rate = DEFAULT_PARK_SECONDS, quota = QUOTA_PARK_SECONDS,
}
local RISK_KINDS = { risk = true, challenge = true, blocked = true }

-- Risk control escalates while an account keeps calling: each block in a
-- row doubles the park, up to an hour. An answered request resets it.
local function park_seconds(credential_entry, kind)
  if not RISK_KINDS[kind] then return PARK_SECONDS[kind] end
  local streak = (llm_router.storage.get(RISK_STREAK_SCOPE, credential_entry.id) or 0) + 1
  llm_router.storage.set(RISK_STREAK_SCOPE, credential_entry.id, streak, { ttl = RISK_STREAK_SECONDS })
  return math.min(RISK_PARK_SECONDS * 2 ^ (streak - 1), MAX_RISK_PARK_SECONDS)
end

-- Tries usable accounts in turn until one answers. output = {sink,
-- delivered, reset}: after the first delivered byte a failure is final.
local function serve(ctx, request, output)
  if type(request.tools) == "table" and #request.tools > 0 then
    return nil, { message = "Qwen Web does not take tools", code = "invalid_request_error", status = 400 }
  end
  local model, search = upstream_model(request.model_name)
  local tried = 0
  local limited = 0
  local last_issue = nil
  for _, credential_entry in ipairs(llm_router.credentials.list()) do
    local account = account_of(credential_entry.data)
    local limit_key = credential_entry.id .. "/" .. model
    if account ~= nil and llm_router.storage.get(MODEL_LIMIT_SCOPE, limit_key) ~= nil then
      limited = limited + 1
    elseif account ~= nil then
      tried = tried + 1
      output.reset()
      local usage, issue = attempt(account, model, search, request, output)
      if issue == nil then
        llm_router.storage.delete(RISK_STREAK_SCOPE, credential_entry.id)
        return usage, nil
      end
      last_issue = issue
      if output.delivered() then return nil, call_error(issue) end
      if issue.kind == "model_limit" then
        -- The account keeps serving other models.
        limited = limited + 1
        llm_router.storage.set(MODEL_LIMIT_SCOPE, limit_key, true, { ttl = QUOTA_PARK_SECONDS })
        llm_router.log("warn", "model limit reached", { cred_id = credential_entry.id, model = model, wait = QUOTA_PARK_SECONDS })
      elseif PARK_SECONDS[issue.kind] ~= nil then
        park(credential_entry, park_seconds(credential_entry, issue.kind), issue.kind, issue.message)
      elseif issue.kind == "invalid_credentials" or issue.kind == "auth" then
        llm_router.log("warn", "account rejected", { cred_id = credential_entry.id, error = issue.message })
        retire(ctx, credential_entry, "invalid_credentials")
      else
        return nil, call_error(issue)
      end
    end
  end
  -- list() skips parked accounts, so an empty list means either none is
  -- signed in or every one waits out a park.
  if tried == 0 and limited > 0 then
    return nil, { message = "every Qwen account reached today's limit for " .. model .. ": pick another model", code = "rate_limit", status = 429 }
  end
  if tried == 0 then
    return nil, { message = "no Qwen account is available: sign one in or wait until a parked one returns", code = "rate_limit", status = 429 }
  end
  local reason = last_issue.kind == "risk" and "Qwen risk control blocked the request" or last_issue.message
  return nil, { message = "no Qwen account accepted the request: " .. reason, code = "rate_limit", status = 429 }
end

local function usage_of(raw)
  if type(raw) ~= "table" then return nil end
  local prompt = raw.input_tokens or raw.prompt_tokens or 0
  local completion = raw.output_tokens or raw.completion_tokens or 0
  return {
    prompt_tokens = prompt,
    completion_tokens = completion,
    total_tokens = raw.total_tokens or (prompt + completion),
    prompt_tokens_details = { cached_tokens = (raw.prompt_tokens_details or {}).cached_tokens or 0 },
    completion_tokens_details = { reasoning_tokens = (raw.output_tokens_details or {}).reasoning_tokens or 0 },
  }
end

local function complete(ctx, request)
  local content, reasoning = {}, {}
  local raw_usage, fault = serve(ctx, request, {
    sink = function(kind, text)
      if kind == "reasoning" then table.insert(reasoning, text) else table.insert(content, text) end
    end,
    delivered = function() return false end,
    reset = function() content, reasoning = {}, {} end,
  })
  if fault ~= nil then return nil, fault end
  local message = { role = "assistant", content = table.concat(content) }
  if #reasoning > 0 then message.reasoning_content = table.concat(reasoning) end
  return {
    id = "chatcmpl-" .. llm_router.random_hex(12), object = "chat.completion", created = os.time(),
    choices = { { index = 0, message = message, finish_reason = "stop" } },
    usage = usage_of(raw_usage),
  }, nil
end

local function complete_stream(ctx, request, emit)
  local id = "chatcmpl-" .. llm_router.random_hex(12)
  local created = os.time()
  local started = false
  local function send(delta, finish_reason)
    if not started then
      delta.role = "assistant"
      started = true
    end
    emit({ id = id, object = "chat.completion.chunk", created = created,
      choices = { { index = 0, delta = delta, finish_reason = finish_reason } } })
  end
  local raw_usage, fault = serve(ctx, request, {
    sink = function(kind, text)
      if kind == "reasoning" then send({ reasoning_content = text }, nil) else send({ content = text }, nil) end
    end,
    delivered = function() return started end,
    reset = function() end,
  })
  if fault ~= nil then return nil, fault end
  send({}, "stop")
  local usage = usage_of(raw_usage)
  if usage ~= nil then
    emit({ id = id, object = "chat.completion.chunk", created = created, choices = json.array({}), usage = usage })
  end
  return nil, nil
end

-- ── Model cards ──

local function first_account()
  for _, credential_entry in ipairs(llm_router.credentials.list()) do
    local account = account_of(credential_entry.data)
    if account ~= nil then return account end
  end
  return nil
end

local function has_value(list, wanted)
  for _, value in ipairs(list or {}) do
    if value == wanted then return true end
  end
  return false
end

local function model_cards(entry)
  local meta = (entry.info or {}).meta or {}
  local capabilities = meta.capabilities or {}
  local card = {
    name = entry.id,
    display_name = entry.name or entry.id,
    description = meta.short_description,
    context_window = meta.max_context_length,
    input_modalities = { "text" },
    output_modalities = { "text" },
    endpoints = { "chat/completions" },
  }
  if capabilities.thinking == true then
    card.capabilities = { "reasoning" }
    card.reasoning = { supported_efforts = { "high", "medium", "none" }, default_effort = "medium", default_enabled = true }
  end
  local cards = { card }
  if capabilities.search == true then
    local search = {}
    for key, value in pairs(card) do search[key] = value end
    search.name = entry.id .. SEARCH_SUFFIX
    search.display_name = card.display_name .. " Search"
    search.description = "Qwen web chat with web search."
    table.insert(cards, search)
  end
  return cards
end

-- ── Sign-in wizard ──

local FAILURE_KEYS = {
  invalid_credentials = "qwen.error.invalid_credentials",
  blocked = "qwen.error.blocked",
  unavailable = "qwen.error.unavailable",
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
    title = "qwen.sign_in.title",
    subtitle = "qwen.sign_in.subtitle",
    content = {
      { type = "input", name = "email", label = "qwen.field.email", placeholder = "you@example.com", value = values.email, required = true },
      { type = "secret", name = "password", label = "qwen.field.password", required = true },
      { type = "link", text = "qwen.sign_in.open", url = SIGN_IN_URL },
      { type = "button", text = "qwen.sign_in.submit", form_action = "sign_in", variant = "primary" },
    },
  })
  return { render = nodes }
end

-- ── Registration ──

llm_router.register("qwen-web", {
  icon = CHAT_ORIGIN .. "/static/favicon.png",

  initialize = function(ctx)
    local fault = ensure_running(20000)
    if fault ~= nil then
      llm_router.log("info", "container not ready at initialization", { reason = fault.reason, error = fault.message })
    end
  end,

  validate_credentials = function(data)
    local account = account_of(data)
    if account == nil then return false, { message = "set the email and the password hash" } end
    if not string.match(account.password_hash, "^" .. string.rep("%x", 64) .. "$") then
      return false, { message = "the password hash must be 64 hex characters" }
    end
    return true
  end,

  auth_initiate = function(ctx) return sign_in_page() end,

  auth_step = function(ctx, input)
    local values = input.values or {}
    if input.action ~= "sign_in" then return sign_in_page(values) end
    local email = string.lower(trim(values.email))
    if email == "" or type(values.password) ~= "string" or values.password == "" then
      return sign_in_page(values, "qwen.error.missing_fields")
    end
    local password_hash = llm_router.sha256_hex(values.password)
    local session, issue = sign_in(email, password_hash)
    if session == nil then
      llm_router.log("warn", "account sign-in failed", { reason = issue.kind, error = issue.message, operation = "auth_step" })
      return sign_in_page(values, FAILURE_KEYS[issue.kind] or "qwen.error.unavailable", issue.message)
    end
    llm_router.log("info", "account signed in", { operation = "auth_step" })
    return { credentials = { email = email, password_hash = password_hash } }
  end,

  check_health = function(ctx, credential)
    local account = account_of(credential.data)
    if account == nil then return { status = "unhealthy", message = "the credential has no email or password hash" } end
    local token, issue = access_token(account, true)
    if token ~= nil then return { status = "healthy" } end
    if issue.kind == "invalid_credentials" then return { status = "unhealthy", message = issue.message } end
    return { status = "unknown", message = issue.kind .. ": " .. issue.message }
  end,

  complete = complete,
  complete_stream = complete_stream,

  get_model_infos = function(ctx)
    local account = first_account()
    if account == nil then return {}, nil end
    local token, issue = access_token(account, false)
    if token == nil then return nil, call_error(issue) end
    local response, request_issue = page_request(account, token, "GET", "/api/v2/models/", nil)
    if request_issue ~= nil then return nil, call_error(request_issue) end
    local decoded = decode_json(response.body)
    local data = decoded and decoded.data
    local entries = type(data) == "table" and data.data or nil
    if response.status ~= 200 or type(entries) ~= "table" then
      return nil, call_error(classify(response.status, response.body))
    end
    local cards = {}
    for _, entry in ipairs(entries) do
      local info = entry.info or {}
      local meta = info.meta or {}
      if info.is_active ~= false and has_value(meta.chat_type, "t2t") then
        for _, card in ipairs(model_cards(entry)) do table.insert(cards, card) end
      end
    end
    return cards, nil
  end,

  check_model = function(ctx, model_ref)
    if model_ref.endpoint ~= "chat" then
      return { status = "unknown", message = "endpoint not supported " .. tostring(model_ref.endpoint) }
    end
    local result, fault = complete(ctx, {
      model_name = model_ref.model_name,
      messages = { { role = "user", content = "hi" } },
      reasoning_effort = "none",
    })
    if fault ~= nil then
      if fault.code == "rate_limit" then return { status = "quota_exceeded", message = fault.message } end
      if fault.code == "not_found" or fault.code == "upstream_error" then return { status = "unhealthy", message = fault.message } end
      return { status = "unknown", message = fault.message }
    end
    return { status = "healthy", response = string.sub(result.choices[1].message.content, 1, 60) }
  end,

  locales = {
    en = {
      ["qwen.sign_in.title"] = "Sign In to Qwen",
      ["qwen.sign_in.subtitle"] = "Use the email and password of your chat.qwen.ai account.",
      ["qwen.sign_in.open"] = "Open Qwen Chat",
      ["qwen.sign_in.submit"] = "Sign In",
      ["qwen.field.email"] = "Email",
      ["qwen.field.password"] = "Password",
      ["qwen.error.missing_fields"] = "Enter the email and the password.",
      ["qwen.error.invalid_credentials"] = "Qwen rejected the email or the password.",
      ["qwen.error.blocked"] = "Qwen blocked the sign-in from this network. Try again later.",
      ["qwen.error.unavailable"] = "Qwen did not answer. Try again.",
    },
    ru = {
      ["qwen.sign_in.title"] = "Вход в Qwen",
      ["qwen.sign_in.subtitle"] = "Введите email и пароль аккаунта chat.qwen.ai.",
      ["qwen.sign_in.open"] = "Открыть Qwen Chat",
      ["qwen.sign_in.submit"] = "Войти",
      ["qwen.field.email"] = "Email",
      ["qwen.field.password"] = "Пароль",
      ["qwen.error.missing_fields"] = "Введите email и пароль.",
      ["qwen.error.invalid_credentials"] = "Qwen отклонил email или пароль.",
      ["qwen.error.blocked"] = "Qwen заблокировал вход из этой сети. Повторите позже.",
      ["qwen.error.unavailable"] = "Qwen не ответил. Повторите попытку.",
    },
  },
})
