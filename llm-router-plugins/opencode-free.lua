--- @plugin OpenCode Free
--- @author TheSlopMachine
--- @version 6.7.0
--- @plugin_api 1.5
--- @description OpenAI/Anthropic/Google compatible free provider OpenCode Free (no key required)
--- @allow_host opencode.ai
--- @allow_host models.opencode.ai
--- @allow_host registry.npmjs.org
--- @allow_host raw.githubusercontent.com

-- The free tier only serves the genuine CLI, so every fingerprint-critical
-- byte (version, system prompt head, tool descriptions, ids) is reproduced
-- from upstream sources. Sandbox state is fresh per call: module-level
-- locals are per-request, anything that must outlive a request lives in
-- llm_router.storage.

-- Base path follows the model catalog (opencode.api): /zen/v1 now answers
-- FreeTierError, the live path is /inference/v1.
local BASE_URL = "https://opencode.ai/inference/v1"
local MODELS_DEV_URL = "https://models.opencode.ai/api.json"
local NPM_VERSION_URL = "https://registry.npmjs.org/opencode-ai/latest"
-- Prompt and tool-description bytes are pinned to the release tag matching
-- the impersonated CLI version: the dev branch restructured into v2
-- (packages/cli, new system prompt) while released 1.18.x clients still send
-- the v1 shapes, so dev bytes would mismatch the UA. No branch fallback:
-- fail closed instead.
local RAW_BASE = "https://raw.githubusercontent.com/anomalyco/opencode/"

-- The free-tier gate answers 401/403 with this text to anything that does
-- not look like the genuine client.
local FINGERPRINT_MARKER = "only be used from within OpenCode"

-- Generations over large contexts run long; the wide timeout only bounds
-- dead connections, progress streams underneath it.
local GENERATION_TIMEOUT_MS = 300000

-- Free-tier models think inside the output budget, including chat-protocol
-- models with no reasoning flag. Requests below the floor starve the text
-- (reasoning consumes the whole budget), so the floor applies to every
-- model, explicit caller budgets included.
local MIN_OUTPUT_TOKENS = 512
local DEFAULT_OUTPUT_TOKENS = 32000

-- Router log values truncate at 2048 characters; dumps cap just below so the
-- truncation marker (with the original size) survives.
local DUMP_CAP = 1900

-- Upstream verdicts on one attempt: try the next exit or return the error.
local NEXT = "next"
local STOP = "stop"

-- Terminal error table in the router's error contract.
local function fault(message, code, status)
  return { message = message, code = code, status = status }
end

---------------------------------------------------------------------------
-- Storage and JSON helpers
---------------------------------------------------------------------------

-- Storage is a cache layer, never a correctness dependency: a faulting read
-- is a miss and a faulting write is skipped. Both are logged.
local function storage_call(operation, scope, ...)
  local ok, result = pcall(llm_router.storage[operation], scope, ...)
  if ok then return result end
  llm_router.log("warn", "storage operation failed", { operation = operation, reason = tostring(result) })
  return nil
end

local function store_get(scope, key) return storage_call("get", scope, key) end
local function store_set(scope, key, value, ttl) storage_call("set", scope, key, value, { ttl = ttl }) end
local function store_delete(scope, key) storage_call("delete", scope, key) end

-- Diagnostics only: never feeds a wire body or a key.
local function log_json(value)
  local ok, encoded = pcall(json.encode, value)
  if ok and type(encoded) == "string" then return encoded end
  return "<unencodable>"
end

local function encode_json(value, what)
  local ok, encoded = pcall(json.encode, value)
  if ok and type(encoded) == "string" then return encoded end
  return nil, fault("request " .. what .. " cannot be serialized", "invalid_request_error", 400)
end

-- Empty Lua tables encode as JSON objects, but the upstream validator needs
-- real empty arrays (reasoning `summary`, tool `required`). Payloads carry
-- this sentinel string where an empty array belongs, and the serialized body
-- swaps exactly the quoted sentinel token. User text cannot collide: quotes
-- inside JSON strings are escaped, so a whole quoted token never appears.
local WIRE_EMPTY_ARRAY = "__wire_empty_array__"

local function wire_body(payload)
  local body, err = encode_json(payload, "payload")
  if body == nil then return nil, err end
  return (body:gsub('"' .. WIRE_EMPTY_ARRAY .. '"', "[]"))
end

-- Router-bound values need a table that serializes as an empty array.
local function empty_array() return json.decode("[]") end

local function capped(value)
  local s = tostring(value or "")
  if #s > DUMP_CAP then return s:sub(1, DUMP_CAP) .. "\n[TRUNCATED total=" .. tostring(#s) .. "]" end
  return s
end

local function redacted_headers(headers)
  local copy = {}
  for k, v in pairs(headers) do
    if tostring(k):lower() == "authorization" then
      copy[k] = "{redacted}"
    else
      copy[k] = v
    end
  end
  return log_json(copy)
end

local function dump_request(job)
  llm_router.log("debug", "request dump", {
    endpoint = job.protocol.path,
    headers = redacted_headers(job.headers),
    body = capped(job.body),
    event = "request_dump",
  })
end

local function dump_response(resp, model_name)
  local headers = ""
  if type(resp.headers) == "table" then headers = log_json(resp.headers) end
  llm_router.log("debug", "response dump", {
    model = model_name,
    status = resp.status,
    headers = headers,
    body = capped(resp.body),
    event = "response_dump",
  })
end

---------------------------------------------------------------------------
-- Remote text: CLI version, prompt heads, tool descriptions
---------------------------------------------------------------------------

local TEXT_CACHE_SCOPE = "remote_text"

local function http_get_text(url)
  local client = llm_router.http_client({ timeout_ms = 30000 })
  local resp, err = client:request({ method = "GET", url = url, headers = {} })
  if err ~= nil or resp.status ~= 200 or type(resp.body) ~= "string" or resp.body == "" then
    llm_router.log("warn", "remote text fetch failed", {
      detail = url,
      status = resp and resp.status,
      reason = err and err.message,
    })
    return nil
  end
  return resp.body
end

-- CLI version for the User-Agent, from the npm registry metadata the
-- official upgrade path prefers over the rate-limited GitHub API:
-- /latest answers {"version": "1.18.35", ...} with no auth. The fresh copy
-- refreshes daily; the last known copy outlives it so a registry outage does
-- not stop traffic. With neither, the request fails closed: a guessed
-- version would put a wrong fingerprint on the wire.
local VERSION_TTL = 86400
local VERSION_LAST_KNOWN_TTL = 30 * 86400

local function valid_version(value)
  return type(value) == "string" and value:match("^%d+%.%d+%.%d+$") ~= nil
end

local function opencode_version()
  local fresh = store_get(TEXT_CACHE_SCOPE, "cli_version")
  if valid_version(fresh) then return fresh end
  local body = http_get_text(NPM_VERSION_URL)
  if body ~= nil then
    local ok, parsed = pcall(json.decode, body)
    if ok and type(parsed) == "table" and valid_version(parsed.version) then
      store_set(TEXT_CACHE_SCOPE, "cli_version", parsed.version, VERSION_TTL)
      store_set(TEXT_CACHE_SCOPE, "cli_version_last", parsed.version, VERSION_LAST_KNOWN_TTL)
      return parsed.version
    end
  end
  local last = store_get(TEXT_CACHE_SCOPE, "cli_version_last")
  if valid_version(last) then
    llm_router.log("warn", "serving last known client version")
    return last
  end
  return nil, fault("opencode client version unavailable", "server_error", 502)
end

local function user_agent(version) return "opencode/" .. version end

local function catalog_user_agent(version) return "opencode/latest/" .. version .. "/cli" end

-- Repo text is pinned to a release tag, so the bytes behind a version key
-- never change: the TTL only bounds how long unused versions linger. A fetch
-- failure with a cold cache fails closed, since a wrong fingerprint is worse
-- than an error.
local REPO_TEXT_TTL = 7 * 86400
local PROMPT_DIR = "packages/opencode/src/session/prompt/"
local TOOL_DESC_DIR = "packages/opencode/src/tool/"

local function fetch_repo_text(version, path)
  local key = version .. ":" .. path
  local cached = store_get(TEXT_CACHE_SCOPE, key)
  if type(cached) == "string" and cached ~= "" then return cached end
  local body = http_get_text(RAW_BASE .. "v" .. version .. "/" .. path)
  if body == nil then return nil end
  store_set(TEXT_CACHE_SCOPE, key, body, REPO_TEXT_TTL)
  return body
end

-- SystemPrompt.provider owns the family mapping; the same text serves both
-- protocols, only the role wrapper differs.
local function prompt_file_for_model(model_name)
  local m = model_name:lower()
  if m:find("muse", 1, true) then return "meta.txt", true end
  if m:find("gpt-4", 1, true) or m:find("o1", 1, true) or m:find("o3", 1, true) then return "beast.txt", false end
  if m:find("gpt", 1, true) then
    if m:find("gpt-6", 1, true) then return "gpt-astra.txt", false end
    if m:find("codex", 1, true) then return "codex.txt", false end
    return "gpt.txt", false
  end
  if m:find("gemini-", 1, true) then return "gemini.txt", false end
  if m:find("claude", 1, true) then return "anthropic.txt", false end
  if m:find("trinity", 1, true) then return "trinity.txt", false end
  if m:find("kimi", 1, true) then return "kimi.txt", false end
  return "default.txt", false
end

-- Head text only: the live tail (working directory, skills, MCP
-- instructions) is machine-specific and unreproducible, so it is omitted.
local function agent_prompt(version, model_name)
  local file, is_meta = prompt_file_for_model(model_name)
  local head = fetch_repo_text(version, PROMPT_DIR .. file)
  if head == nil then
    return nil, fault("agent prompt unavailable (" .. file .. ")", "server_error", 502)
  end
  if is_meta then
    local name = "Muse Spark"
    if model_name:lower():find("muse-glimmer", 1, true) then name = "Muse Glimmer" end
    head = head:gsub("{{MODEL_NAME}}", name)
  end
  return head
end

-- pwsh/win32 render of tool/shell/shell.txt, ported from
-- tool/shell/prompt.ts (powershellCommandSection + chainGuidance PS branch,
-- limits 2000 lines / 51200 bytes, 120000ms default timeout).
-- The tmp path is machine-specific upstream; this value reproduces the
-- proven Windows-client bytes and is documented, not derived.
local SHELL_TMP_DIR = "C:\\Users\\Thinker\\AppData\\Local\\Temp\\opencode"
local SHELL_INTRO = "Executes a given PowerShell (7+) command with optional timeout, ensuring proper handling and security measures."

local SHELL_CHAIN_GUIDANCE = "If the commands depend on each other and must run sequentially, use a single bash tool call with '&&' to chain them together (e.g., `git add . && git commit -m \"message\" && git push`). For instance, if one operation must complete before another starts (like New-Item before Copy-Item, Write before bash for git operations, or git add before git commit), run these operations sequentially instead."
local SHELL_NOTES = "# PowerShell (7+) shell notes\n"
  .. "- This cross-platform shell supports pipeline chain operators (`&&` and `||`).\n"
  .. "- Use double quotes for interpolated strings (`\"Hello $name\"`), single quotes for verbatim strings.\n"
  .. "- Prefer full cmdlet names like `Get-ChildItem`, `Set-Content`, `Remove-Item`, and `New-Item` over aliases.\n"
  .. "- Use `$(...)` for subexpressions. Use `@(...)` for array expressions.\n"
  .. "- To call a native executable whose path contains spaces, use the call operator: `& \"path/to/exe\" args`.\n"
  .. "- Escape special characters with the PowerShell backtick character."
local SHELL_COMMAND_SECTION = SHELL_NOTES
  .. "\n\nBefore executing the command, please follow these steps:\n\n"
  .. "1. Directory Verification:\n"
  .. "   - If the command will create new directories or files, first use `Test-Path -LiteralPath <parent>` to verify the parent directory exists and is the correct location\n"
  .. "   - For example, before creating `foo\\bar`, first use `Test-Path -LiteralPath \"foo\"` to check that `foo` exists and is the intended parent directory\n"
  .. "\n2. Command Execution:\n"
  .. "   - Always quote file paths that contain spaces with double quotes (e.g., Remove-Item -LiteralPath \"path with spaces\\file.txt\")\n"
  .. "   - Examples of proper quoting:\n"
  .. "     - New-Item -ItemType Directory -Path \"My Documents\" (correct)\n"
  .. "     - New-Item -ItemType Directory -Path My Documents (incorrect - path is split)\n"
  .. "     - & \"path with spaces\\script.ps1\" (correct)\n"
  .. "     - path with spaces\\script.ps1 (incorrect - path is split and not invoked)\n"
  .. "   - After ensuring proper quoting, execute the command.\n"
  .. "   - Capture the output of the command.\n"
  .. "\nUsage notes:\n"
  .. "   - The command argument is required.\n"
  .. "   - You can specify an optional timeout in milliseconds. If not specified, commands will time out after 120000ms.\n"
  .. "   - If the output exceeds 2000 lines or 51200 bytes, it will be truncated and the full output will be written to a file. You can use Read with offset/limit to read specific sections or Grep to search the full content. Do NOT use `Select-Object -First`, `Select-Object -Last`, or other truncation commands to limit output; the full output will already be captured to a file for more precise searching.\n"
  .. "\n   - Avoid using Shell with PowerShell file/content cmdlets unless explicitly instructed or when these cmdlets are truly necessary for the task. Instead, always prefer using the dedicated tools for these commands:\n"
  .. "     - File search: Use Glob (NOT Get-ChildItem)\n"
  .. "     - Content search: Use Grep (NOT Select-String)\n"
  .. "     - Read files: Use Read (NOT Get-Content)\n"
  .. "     - Edit files: Use Edit (NOT Set-Content)\n"
  .. "     - Write files: Use Write (NOT Set-Content/Out-File or here-strings)\n"
  .. "     - Communication: Output text directly (NOT Write-Output/Write-Host)\n"
  .. "   - When issuing multiple commands:\n"
  .. "     - If the commands are independent and can run in parallel, make multiple bash tool calls in a single message. For example, if you need to run \"git status\" and \"git diff\", send a single message with two bash tool calls in parallel.\n"
  .. "     - "
  .. SHELL_CHAIN_GUIDANCE
  .. "\n     - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail\n"
  .. "     - DO NOT use newlines to separate commands (newlines are ok in quoted strings)\n"
  .. "   - AVOID changing directories inside the command. Use the `workdir` parameter to change directories instead.\n"
  .. "     <good-example>\n"
  .. "     Use workdir=\"project\\subdir\" with command: pytest tests\n"
  .. "     </good-example>\n"
  .. "     <bad-example>\n"
  .. "     Set-Location -LiteralPath \"project\\subdir\" && pytest tests\n"
  .. "     </bad-example>"
local SHELL_WORKDIR_SECTION = "All commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID changing directories inside the command - use `workdir` instead."

local function shell_description(version)
  local template = fetch_repo_text(version, TOOL_DESC_DIR .. "shell/shell.txt")
  if template == nil then return nil end
  local values = {
    intro = SHELL_INTRO,
    os = "win32",
    shell = "pwsh",
    tmp = SHELL_TMP_DIR,
    workdirSection = SHELL_WORKDIR_SECTION,
    commandSection = SHELL_COMMAND_SECTION,
  }
  -- Single pass with a function replacement: substituted text is never
  -- rescanned and `%` in it needs no escaping. Unknown placeholders stay.
  return (template:gsub("%${(%w+)}", function(name) return values[name] end))
end

-- Tool description prose comes from the tool sibling .txt files. Parameter
-- schemas below are structural (TS-annotation-sourced) and pinned against
-- live captures, not prose.
local TOOL_DESC_FILES = {
  read = "read.txt",
  write = "write.txt",
  edit = "edit.txt",
  glob = "glob.txt",
  grep = "grep.txt",
}

local SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
local HUGE_INT = 9007199254740991

local TOOL_PARAMS = {
  bash = {
    ["$schema"] = SCHEMA_DRAFT,
    type = "object",
    properties = {
      command = { type = "string", description = "The command to execute" },
      timeout = { minimum = -HUGE_INT, exclusiveMinimum = 0, type = "integer", maximum = HUGE_INT },
      workdir = { type = "string" },
    },
    required = { "command" },
  },
  read = {
    ["$schema"] = SCHEMA_DRAFT,
    type = "object",
    properties = {
      filePath = { type = "string", description = "The absolute path to the file or directory to read" },
      offset = { minimum = 0, type = "integer", maximum = HUGE_INT },
      limit = { minimum = 0, type = "integer", maximum = HUGE_INT },
    },
    required = { "filePath" },
  },
  write = {
    ["$schema"] = SCHEMA_DRAFT,
    type = "object",
    properties = {
      content = { type = "string", description = "The content to write to the file" },
      filePath = {
        type = "string",
        description = "The absolute path to the file to write (must be absolute, not relative)",
      },
    },
    required = { "content", "filePath" },
  },
  edit = {
    ["$schema"] = SCHEMA_DRAFT,
    type = "object",
    properties = {
      filePath = { type = "string", description = "The absolute path to the file to modify" },
      oldString = { type = "string", description = "The text to replace" },
      newString = { type = "string", description = "The text to replace it with (must be different from oldString)" },
      replaceAll = { type = "boolean", description = "Replace all occurrences of oldString (default false)" },
    },
    required = { "filePath", "oldString", "newString" },
  },
  glob = {
    ["$schema"] = SCHEMA_DRAFT,
    type = "object",
    properties = {
      pattern = { type = "string", description = "The glob pattern to match files against" },
      path = { type = "string", description = "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided." },
    },
    required = { "pattern" },
  },
  grep = {
    ["$schema"] = SCHEMA_DRAFT,
    type = "object",
    properties = {
      pattern = { type = "string", description = "The regex pattern to search for in file contents" },
      path = { type = "string", description = "The directory to search in. Defaults to the current working directory." },
      include = { type = "string", description = 'File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")' },
    },
    required = { "pattern" },
  },
}

---------------------------------------------------------------------------
-- Tools
---------------------------------------------------------------------------

local function tool_name(t)
  local fn = t["function"]
  if type(fn) == "table" and type(fn.name) == "string" and fn.name ~= "" then return fn.name end
  if type(t.name) == "string" then return t.name end
  return ""
end

-- Tool definitions the caller sent. The router hands over an array.
local function client_tools(tools)
  local out = {}
  if type(tools) == "table" then
    for _, t in ipairs(tools) do
      if type(t) == "table" then table.insert(out, t) end
    end
  end
  return out
end

local INJECTED_TOOL_NAMES = { "bash", "read", "write", "edit", "glob", "grep" }

-- Genuine main turns always carry tools: the six built-ins join every
-- request (descriptions fetched, schemas structural). Returns (nil, err)
-- when a description is unavailable.
local function injected_tools(version)
  local out = {}
  for _, name in ipairs(INJECTED_TOOL_NAMES) do
    local desc
    if name == "bash" then
      desc = shell_description(version)
    else
      desc = fetch_repo_text(version, TOOL_DESC_DIR .. TOOL_DESC_FILES[name])
    end
    if desc == nil or desc == "" then
      return nil, fault("tool description unavailable (" .. name .. ")", "server_error", 502)
    end
    table.insert(out, {
      type = "function",
      ["function"] = { name = name, description = desc, parameters = TOOL_PARAMS[name] },
    })
  end
  return out
end

-- Client tools first, then every built-in the client did not define itself.
local function chat_tools(version, request)
  local out, seen = {}, {}
  for _, t in ipairs(client_tools(request.tools)) do
    seen[tool_name(t)] = true
    table.insert(out, t)
  end
  local builtins, err = injected_tools(version)
  if builtins == nil then return nil, err end
  for _, t in ipairs(builtins) do
    if not seen[tool_name(t)] then table.insert(out, t) end
  end
  return out
end

-- Chat- or Responses-shaped definition to a Responses function tool.
-- Unnamed definitions (hosted tools) have no Responses equivalent.
local function to_responses_tool(t)
  local name = tool_name(t)
  if name == "" then return nil end
  local fn = type(t["function"]) == "table" and t["function"] or {}
  local description = fn.description
  if type(description) ~= "string" or description == "" then description = t.description end
  local parameters = fn.parameters or t.parameters
  if parameters == nil then
    parameters = { type = "object", properties = {}, required = WIRE_EMPTY_ARRAY }
  end
  return { type = "function", name = name, description = description or "", parameters = parameters }
end

local function responses_tools(version, request)
  local out, seen = {}, {}
  for _, t in ipairs(client_tools(request.tools)) do
    local converted = to_responses_tool(t)
    if converted ~= nil then
      seen[converted.name] = true
      table.insert(out, converted)
    end
  end
  local builtins, err = injected_tools(version)
  if builtins == nil then return nil, err end
  for _, t in ipairs(builtins) do
    local fn = t["function"]
    if not seen[fn.name] then
      table.insert(out, { type = "function", name = fn.name, description = fn.description, parameters = fn.parameters })
    end
  end
  return out
end

local function output_token_budget(request)
  local value = tonumber(request.max_completion_tokens)
  if value == nil or value <= 0 then value = tonumber(request.max_tokens) end
  if value == nil or value <= 0 then return DEFAULT_OUTPUT_TOKENS end
  return math.max(value, MIN_OUTPUT_TOKENS)
end

---------------------------------------------------------------------------
-- ULID-shaped ids and anonymous headers
---------------------------------------------------------------------------

local ULID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
local HEX_DIGITS = "0123456789abcdef"

local function hex12(n)
  local out = {}
  for i = 12, 1, -1 do
    local d = n % 16
    out[i] = HEX_DIGITS:sub(d + 1, d + 1)
    n = (n - d) / 16
  end
  return table.concat(out)
end

local function time_head(descending, ms, counter)
  local cur = ms * 4096 + counter
  if descending then cur = 281474976710655 - cur end
  return hex12(cur)
end

local function ulid_tail()
  local hex = llm_router.random_hex(14)
  local out = {}
  for i = 1, 14 do
    local c = tonumber(hex:sub(i * 2 - 1, i * 2), 16) % 62
    out[i] = ULID_CHARS:sub(c + 1, c + 1)
  end
  return table.concat(out)
end

-- The time head uses the current second plus a uniform sub-second part,
-- matching the genuine millisecond distribution.
local function mint_ms()
  local jitter = tonumber(llm_router.random_hex(2):sub(1, 4), 16)
  return os.time() * 1000 + (jitter % 1000)
end

local function mint_session(ms) return "ses_" .. time_head(true, ms, 1) .. ulid_tail() end

local function mint_msg(ms) return "msg_" .. time_head(false, ms, 2) .. ulid_tail() end

-- Free tier is anonymous: every request carries Bearer public, the
-- ULID-shaped session pair, and the run-scoped request id.
local function opencode_headers(version, ses, msg)
  return {
    ["User-Agent"] = user_agent(version),
    ["Content-Type"] = "application/json",
    ["Accept"] = "application/json",
    ["Authorization"] = "Bearer public",
    ["x-opencode-client"] = "cli",
    ["x-opencode-project"] = "global",
    ["x-opencode-session"] = ses,
    ["x-opencode-session-id"] = ses,
    ["x-opencode-request"] = msg,
  }
end

---------------------------------------------------------------------------
-- Conversation identity: session id and reasoning blobs
---------------------------------------------------------------------------

-- Every request carries the full history, so the conversation it continues
-- is the longest earlier dialogue that is a prefix of its messages. Dialogue
-- keys chain over the messages: keys[i] identifies messages[1..i] (model
-- scoped, so identical dialogue on different models never shares state), and
-- one pass over the history yields every prefix key.
--
-- The gap between two requests of one conversation is not fixed: a plain
-- turn appends an assistant reply plus a user message, a tool round appends
-- the assistant message plus one result per parallel call. So a lookup scans
-- prefixes from the longest down instead of assuming a fixed distance.
local DLG_NS = "9e4f1a2b-3c5d-4e6f-8a9b-0c1d2e3f4a5b"
-- Messages appended between consecutive requests of one conversation stay far
-- below this; it only bounds the scan on a miss (forks, compacted history).
local LOOKBACK = 64

local function dialogue_keys(request, model_name)
  local keys = {}
  if type(request.messages) ~= "table" then return keys end
  local chain = model_name
  for i, message in ipairs(request.messages) do
    local encoded, err = encode_json(message, "message")
    if encoded == nil then return nil, err end
    chain = llm_router.uuid_v5(DLG_NS, chain .. "|" .. encoded)
    keys[i] = model_name .. "|dlg_" .. chain
  end
  return keys
end

local function is_filled_string(value) return type(value) == "string" and value ~= "" end

local function is_filled_array(value) return type(value) == "table" and #value > 0 end

-- Longest prefix length in [newest - LOOKBACK + 1, newest] with an accepted
-- stored value, plus that value. (nil) when none.
local function find_prefix(scope, keys, newest, accept)
  for n = newest, math.max(newest - LOOKBACK + 1, 1), -1 do
    local value = store_get(scope, keys[n])
    if accept(value) then return n, value end
  end
  return nil
end

-- One session id follows a conversation. Each request moves the entry from
-- the prefix it continued to its own full dialogue (10-minute TTL). A request
-- identical to a stored dialogue (client retry after an error) keeps the
-- entry and reuses the id; a fork finds the entry already moved and starts
-- a new session. The id doubles as prompt_cache_key.
local SESSION_SCOPE = "session_v3"
local SESSION_TTL = 600

local function resolve_session(keys)
  local count = #keys
  local n, ses = find_prefix(SESSION_SCOPE, keys, count, is_filled_string)
  if n == nil then
    ses = mint_session(mint_ms())
    llm_router.log("debug", "session missed", { messages = count })
  else
    if n < count then store_delete(SESSION_SCOPE, keys[n]) end
    llm_router.log("debug", "session loaded", { messages = count, count = count - n })
  end
  if count > 0 then store_set(SESSION_SCOPE, keys[count], ses, SESSION_TTL) end
  return ses
end

-- Reasoning blobs are not keyed by dialogue position: see "Reasoning replay"
-- in the payload section. They belong to the assistant turn that made them.

local function reasoning_blobs(response)
  local blobs = {}
  if type(response.output) ~= "table" then return blobs end
  for _, item in ipairs(response.output) do
    if type(item) == "table" and item.type == "reasoning" and is_filled_string(item.encrypted_content) then
      table.insert(blobs, { id = item.id, blob = item.encrypted_content })
    end
  end
  return blobs
end

---------------------------------------------------------------------------
-- Free-tier identity
---------------------------------------------------------------------------

-- Free-tier identity: -free suffix or the known keyless set. Unknown
-- non-suffixed ids default to premium (fail-safe). Proven-broken free ids
-- stay listed nowhere: remove them here when upstream heals.
-- 2026-09-30: jev-1.13-free (upstream 502s), deepseek-v4-flash-free and
-- ling-3.0-flash-fin-free (fatal 400s).
-- 2026-10-10: exo-free (upstream 410: model removed).
local KNOWN_FREE = { ["big-pickle"] = true }
local EXCLUDED_FREE = {
  ["jev-1.13-free"] = true,
  ["deepseek-v4-flash-free"] = true,
  ["ling-3.0-flash-fin-free"] = true,
  ["exo-free"] = true,
}

local function is_free_model(id)
  if type(id) ~= "string" or id == "" then return false end
  if EXCLUDED_FREE[id] then return false end
  if id:sub(-5) == "-free" then return true end
  return KNOWN_FREE[id] == true
end

---------------------------------------------------------------------------
-- Exit selection and rate-limit memory
---------------------------------------------------------------------------

-- Upstream limits one model on one exit, not the exit as a whole: a 429 marks
-- that exit + model pair in plugin storage for the upstream retry horizon
-- (six hours without one). Selection skips a marked pair for that model only,
-- so the same exit keeps serving every other model. Once the unmarked exits are
-- spent, the soonest-expiring marked pairs are retried (see exit_stream). Exit-
-- wide marks (403, 5xx, transport faults) live in RL_SCOPE and are retried
-- last, after the pair-parked exits. Its name is not the 6.6.x "exit_limited": storage survives updates,
-- and those old rows hold whole-exit 429 marks that would block every model.
-- Under the new name they are never read and expire on their own TTL. Expired
-- rows read as missing. Storage faults fail open: an unmarked pair only costs
-- one extra attempt.
local RL_SCOPE = "exit_blocked"
local RL_MODEL_SCOPE = "exit_model_limited"
local RL_TTL = 6 * 3600
local RL_NS = "6d3c2b0e-8f55-4c0a-9d4e-1b7a5e2f9c31"
-- A 403 marks the exit for an hour: blocked egress addresses recover sooner
-- than rate-limited ones.
local FORBIDDEN_TTL = 3600
-- A 5xx or transport break marks the exit briefly: flaky egress recovers
-- faster than blocked or limited addresses.
local SERVER_ERROR_TTL = 600
-- Per-fetch width: one pool query returns up to this many candidates, and
-- the exit iterator yields them one by one before fetching past marked heads.
local MAX_PROXY_CANDIDATES = 30
-- Transient upstream faults: the exit may be flaky while upstream stays
-- healthy.
local TRANSIENT_STATUS = { [500] = true, [502] = true, [503] = true, [504] = true }

-- The direct connection: no proxy_url.
local DIRECT_EXIT = {}

local function is_direct(px) return (px.url or "") == "" end

-- Keys hash the proxy URL so embedded credentials never land in storage.
local function exit_key(px)
  if is_direct(px) then return "direct" end
  return llm_router.uuid_v5(RL_NS, px.url)
end

-- Pair key for model-scoped marks: the model name joins the proxy URL in the
-- hash input, so each exit holds one independent mark per upstream model. The
-- URL stays hashed, so embedded credentials never land in storage.
local function pair_key(px, model_name)
  return llm_router.uuid_v5(RL_NS, px.url .. "\n" .. model_name)
end

-- The direct exit is never parked, model-scoped marks included: without a pool
-- it is the only egress, so a parked pair would fail that model for the whole
-- upstream horizon with no fallback.
local BLOCKED, PARKED = "blocked", "parked"

-- Why an exit is out of rotation for one model: nil when it is usable, BLOCKED
-- for an exit-wide mark, or PARKED with the pair row when only its 429 mark is
-- live.
local function exit_mark(px, model_name)
  if is_direct(px) then return nil end
  local blocked = store_get(RL_SCOPE, exit_key(px))
  if blocked ~= nil then return BLOCKED, blocked end
  local row = store_get(RL_MODEL_SCOPE, pair_key(px, model_name))
  if row ~= nil then return PARKED, row end
  return nil
end

local function mark_exit(px, reason, ttl)
  if is_direct(px) then return end
  local now = os.time()
  store_set(RL_SCOPE, exit_key(px), { at = now, reason = reason, ["until"] = now + ttl }, ttl)
end

-- A 429 parks the exit for this upstream model only. The row keeps its expiry
-- as `until` (epoch seconds), so parked retries can run soonest-expiring first.
local function mark_exit_model(px, model_name, reason, ttl)
  if is_direct(px) then return end
  local now = os.time()
  store_set(RL_MODEL_SCOPE, pair_key(px, model_name), {
    at = now, reason = reason, model = model_name, ["until"] = now + ttl,
  }, ttl)
end

-- A parked retry that answers cleanly is back in rotation: its pair mark goes
-- at once instead of living out the TTL.
local function clear_pair(px, model_name)
  if is_direct(px) then return end
  store_delete(RL_MODEL_SCOPE, pair_key(px, model_name))
end

-- A clean answer from an exit-wide-blocked exit lifts its block.
local function clear_exit(px)
  if is_direct(px) then return end
  store_delete(RL_SCOPE, exit_key(px))
end

-- Transport breaks bind to the exit, not the request. Marks are short so
-- future selections skip the flaky egress without hiding an upstream-wide
-- outage. Cancellation never marks: without transport details the failure
-- carries no exit blame.
local function mark_exit_transport(px, err)
  if err.transport ~= true or is_direct(px) then return end
  mark_exit(px, "server_error", SERVER_ERROR_TTL)
  llm_router.log("warn", "exit transport failed", {
    code = "server_error",
    reason = tostring(err.reason or "network"),
    exit = exit_key(px),
  })
end

-- Most last-resort retries one request may make. The last-resort tier holds the
-- pair-parked exits and the exit-wide-blocked exits, soonest expiry first. This
-- cap bounds the extra upstream calls a limited model can cost per request.
local LAST_RESORT_CAP = 10

-- Lazy exit iterator, in phases. Pool: yields unmarked exits one per call,
-- fetching past marked pool heads on demand; marked exits are collected for the
-- last-resort tier. Direct: once the pool is exhausted, the direct fallback is
-- yielded (never marked). Last resort: the soonest-expiring marked exits, pair
-- parks and exit-wide blocks alike, at most LAST_RESORT_CAP, each yielded with
-- its kind ("parked" or "blocked"). Returns the next function plus a counts
-- table carrying skipped, seen (every exit fetched), model_name, and cancelled.
-- Ends (nil) when every phase is spent, or at once when the request is
-- cancelled: cancellation never degrades to a direct hit or a last-resort retry.
-- The pool search waits for live exits on the first fetch only; later fetches
-- never wait.
local function exit_stream(provider_config, job)
  local pool = nil
  if type(provider_config) == "table" and type(provider_config.proxy) == "table" then
    local configured = provider_config.proxy.pool
    if type(configured) == "string" and configured ~= "" then pool = configured end
  end
  local counts = { skipped = 0, seen = {}, model_name = job.model_name, cancelled = false }
  local seen, pending, last_resort, retries = {}, {}, {}, {}
  local first_fetch, phase = true, "pool"
  local function next_exit()
    while true do
      if #pending > 0 then return table.remove(pending, 1), nil end
      if phase == "pool" then
        local opts = { pool = pool, exclude = seen, limit = MAX_PROXY_CANDIDATES, fallback = "direct" }
        if not first_fetch then opts.timeout_ms = 1 end
        first_fetch = false
        local res = llm_router.proxies.require(opts)
        if res == nil then
          -- Cancelled: no direct leg and no retries.
          counts.cancelled = true
          phase = "done"
          return nil
        end
        if #res.proxies == 0 then
          phase = "direct"
        else
          local usable = 0
          for _, px in ipairs(res.proxies) do
            table.insert(seen, px.url)
            table.insert(counts.seen, px)
            local mark, row = exit_mark(px, job.model_name)
            if mark == nil then
              table.insert(pending, px)
              usable = usable + 1
            else
              counts.skipped = counts.skipped + 1
              table.insert(last_resort, {
                px = px,
                kind = mark == PARKED and "parked" or "blocked",
                until_at = tonumber(row["until"]) or 0,
              })
            end
          end
          llm_router.log("debug", "selecting exits", { usable = usable, skipped = counts.skipped, event = "exits" })
        end
      elseif phase == "direct" then
        phase = "retry"
        table.sort(last_resort, function(a, b) return a.until_at < b.until_at end)
        for i = 1, math.min(#last_resort, LAST_RESORT_CAP) do retries[i] = last_resort[i] end
        if #retries > 0 then
          llm_router.log("debug", "retrying last-resort exits", { model = job.full_model, count = #retries, event = "exits" })
        end
        return DIRECT_EXIT, nil
      elseif phase == "retry" then
        local item = table.remove(retries, 1)
        if item == nil then
          phase = "done"
          return nil
        end
        return item.px, item.kind
      else
        return nil
      end
    end
  end
  return next_exit, counts
end

-- Per-request outcome counters for the terminal message.
local function new_run()
  return {
    denied = 0, -- exits answered with 403
    saw_quota = false, -- any 429 classified as quota
    last_failure = nil, -- message of the latest non-limit failure (cut-off stream, 5xx, ...)
    direct_until = nil, -- epoch second the direct leg's own 429 horizon ends
  }
end

-- Fallback retry hint when no limit horizon is known.
local RETRY_AFTER_SECONDS = "30"

-- Soonest expiry among the pair parks this request knows about and the direct
-- leg's own horizon. Exit-wide blocks are left out: the hint is about rate limits.
local function soonest_limit_until(run, exits)
  local soonest = run.direct_until
  for _, px in ipairs(exits.seen) do
    if not is_direct(px) then
      local row = store_get(RL_MODEL_SCOPE, pair_key(px, exits.model_name))
      local until_at = row and tonumber(row["until"])
      if until_at and (soonest == nil or until_at < soonest) then soonest = until_at end
    end
  end
  return soonest
end

-- Terminal error when the attempt loop ends without success. Never fatal:
-- exhaustion always surfaces as a short-retry 429. The retry hint is the soonest
-- moment a limited candidate becomes eligible again, so a client that waits that
-- long reaches an exit that is no longer parked.
local function terminal_error(run, exits, attempted)
  local code = "rate_limit"
  local what = "free tier rate limited"
  if run.saw_quota then
    code = "insufficient_quota"
    what = "free-tier quota busy"
  end
  local why = "tried everything known"
  if exits.cancelled then why = "request cancelled" end
  local extra = ""
  if run.denied > 0 then extra = " (" .. tostring(run.denied) .. " exits denied with 403)" end
  if run.last_failure ~= nil then extra = extra .. " (last failure: " .. run.last_failure .. ")" end
  local wait = RETRY_AFTER_SECONDS
  local soonest = soonest_limit_until(run, exits)
  if soonest ~= nil then wait = tostring(math.max(1, math.ceil(soonest - os.time()))) end
  llm_router.log("error", "exits exhausted", { code = code, attempt = attempted, count = exits.skipped })
  return {
    message = what .. ", retry in ~" .. wait .. " seconds (" .. why .. ")" .. extra,
    code = code,
    status = 429,
    headers = { ["Retry-After"] = wait },
  }
end

---------------------------------------------------------------------------
-- Upstream error mapping
---------------------------------------------------------------------------

-- Upstream retry horizon in seconds, or nil. Prefers retry-after-ms, then
-- retry-after numeric seconds, then the HTTP-date form. Clamped to
-- [60, 86400]; absent or unparseable headers fall back to RL_TTL.
-- Header names arrive lowercased from the HTTP client.
local RETRY_TTL_MIN = 60
local RETRY_TTL_MAX = 86400
local HTTP_MONTHS = {
  jan = 1, feb = 2, mar = 3, apr = 4, may = 5, jun = 6,
  jul = 7, aug = 8, sep = 9, oct = 10, nov = 11, dec = 12,
}

local function clamp_ttl(seconds)
  return math.max(RETRY_TTL_MIN, math.min(RETRY_TTL_MAX, math.floor(seconds)))
end

local function header_number(headers, key)
  local raw = headers[key]
  if type(raw) ~= "string" then return nil end
  return tonumber(raw:match("^%s*(%d+%.?%d*)%s*$"))
end

-- Days since 1970-01-01 of a proleptic Gregorian date.
local function days_from_civil(year, month, day)
  if month <= 2 then year = year - 1 end
  local era = math.floor(year / 400)
  local year_of_era = year - era * 400
  local day_of_year = math.floor((153 * ((month + 9) % 12) + 2) / 5) + day - 1
  local day_of_era = year_of_era * 365 + math.floor(year_of_era / 4) - math.floor(year_of_era / 100) + day_of_year
  return era * 146097 + day_of_era - 719468
end

-- Epoch seconds of an HTTP-date (always GMT). Computed from the calendar
-- date, not os.time{...}, which would read the fields in the local zone.
local function http_date_epoch(raw)
  if type(raw) ~= "string" then return nil end
  local day, mon, year, hh, mm, ss = raw:match("^%a+, (%d+) (%a+) (%d+) (%d+):(%d+):(%d+) GMT$")
  local month = mon and HTTP_MONTHS[mon:lower()]
  if day == nil or month == nil then return nil end
  return days_from_civil(tonumber(year), month, tonumber(day)) * 86400
    + tonumber(hh) * 3600 + tonumber(mm) * 60 + tonumber(ss)
end

local function retry_ttl_seconds(resp)
  local headers = {}
  if type(resp.headers) == "table" then headers = resp.headers end
  local ms = header_number(headers, "retry-after-ms")
  if ms ~= nil and ms > 0 then return clamp_ttl(ms / 1000) end
  local secs = header_number(headers, "retry-after")
  if secs ~= nil and secs > 0 then return clamp_ttl(secs) end
  local epoch = http_date_epoch(headers["retry-after"])
  local now = os.time()
  if epoch ~= nil and epoch > now then return clamp_ttl(epoch - now) end
  return nil
end

local function is_quota_body(body)
  local lower = body:lower()
  return lower:find("freeusagelimiterror", 1, true) ~= nil
    or lower:find("quota", 1, true) ~= nil
    or lower:find("free usage", 1, true) ~= nil
end

-- Free-tier mapping. The tier is anonymous. A 429 binds to the exit + model
-- pair, quota wording included: only that pair is marked for the upstream
-- retry horizon (6-hour fallback), and the request moves to the next exit for
-- the same model. 403, 5xx and transport faults still mark the whole exit.
-- When all exits are limited the terminal 429 reaches the client. Returns
-- (NEXT, err) to try the next exit or (STOP, err) to return err at once. Only
-- ever called for non-200 answers.
local function map_upstream(run, resp, job, px)
  local model_name = job.model_name
  local status = resp.status
  local body = tostring(resp.body or "")
  dump_response(resp, model_name)
  if status == 401 then
    if body:find(FINGERPRINT_MARKER, 1, true) then
      return STOP, fault("free tier rejected the client fingerprint", "server_error", 401)
    end
    -- Paid models answer 401 without an API key binding on the anonymous
    -- tier: fail the request, never mark anything.
    if not is_free_model(model_name) then
      return STOP, fault("model '" .. model_name .. "' is not available on the free tier", "invalid_request_error", 401)
    end
    return STOP, fault("free tier rejected the anonymous credential", "authentication_error", 401)
  end
  if status == 426 then
    return STOP, fault("free-tier client upgrade required", "authentication_error", 426)
  end
  if status == 429 then
    local quota = is_quota_body(body)
    local ttl = retry_ttl_seconds(resp) or RL_TTL
    llm_router.log("debug", "exit rate limited", { ttl = ttl, quota = quota, exit = exit_key(px), model = job.full_model, event = "rate_limit" })
    mark_exit_model(px, model_name, quota and "quota" or "rate_limit", ttl)
    if is_direct(px) then run.direct_until = math.min(run.direct_until or math.huge, os.time() + ttl) end
    if quota then
      run.saw_quota = true
      return NEXT, fault("free-tier quota exhausted", "insufficient_quota", 429)
    end
    return NEXT, fault("free tier rate limited on current exit", "rate_limit", 429)
  end
  if status == 403 then
    -- Blocked egress address: the exit is at fault, not the request.
    run.denied = run.denied + 1
    llm_router.log("warn", "exit forbidden", { exit = exit_key(px), body = body:sub(1, 1000), event = "forbidden" })
    mark_exit(px, "forbidden", FORBIDDEN_TTL)
    return NEXT, fault("free tier denied the current exit", "server_error", 403)
  end
  if status == 400 or status == 404 or status == 422 then
    return STOP, fault("free tier rejected the request with status " .. tostring(status), "invalid_request_error", status)
  end
  if TRANSIENT_STATUS[status] then
    mark_exit(px, "server_error", SERVER_ERROR_TTL)
    llm_router.log("warn", "exit server error", { status = status, code = "server_error", exit = exit_key(px) })
    return NEXT, fault("free tier returned status " .. tostring(status) .. " on current exit", "server_error", status)
  end
  return STOP, fault("free tier returned status " .. tostring(status), "server_error", status)
end

---------------------------------------------------------------------------
-- Dynamic model catalog
---------------------------------------------------------------------------

-- Zen /models supplies the membership whitelist (ids only);
-- models.opencode.ai/api.json enriches whitelisted ids with limits,
-- modalities and reasoning flags, and enrichment for ids outside the
-- whitelist is discarded. Rows cache in plugin storage for a day: a miss or
-- an expired entry forces a refresh, and any fetch or parse failure serves
-- an empty list, never stale rows.
local MODEL_CACHE_SCOPE = "model_catalog"
local MODEL_CACHE_TTL = 86400
-- Router-required constants the catalog does not provide, plus the
-- conservative defaults for whitelisted ids the catalog does not enrich.
local MODEL_RPM = 60
local MODEL_TPM = 100000
local MODEL_RPD = 500
local MODEL_DEFAULT_CONTEXT_WINDOW = 200000
local MODEL_DEFAULT_MAX_TOKENS = 32000

local function decode_json_table(text)
  local ok, value = pcall(json.decode, text)
  if ok and type(value) == "table" then return value end
  return nil
end

local function string_list(value, default)
  if type(value) ~= "table" then return default end
  local out = {}
  for _, item in ipairs(value) do
    if type(item) == "string" and item ~= "" then table.insert(out, item) end
  end
  if #out == 0 then return default end
  return out
end

-- One router-shaped row from a whitelisted id plus its enrichment entry
-- (nil when the catalog lacks the id, which keeps the safe defaults).
local function build_model_row(name, entry)
  if type(entry) ~= "table" then entry = {} end
  local limit = type(entry.limit) == "table" and entry.limit or {}
  local modalities = type(entry.modalities) == "table" and entry.modalities or {}
  local context_window = math.floor(tonumber(limit.context) or 0)
  if context_window <= 0 then context_window = MODEL_DEFAULT_CONTEXT_WINDOW end
  local max_tokens = math.floor(tonumber(limit.output) or 0)
  if max_tokens <= 0 then max_tokens = MODEL_DEFAULT_MAX_TOKENS end
  local row = {
    name = name,
    display_name = name,
    context_window = context_window,
    max_tokens = max_tokens,
    rpm = MODEL_RPM,
    tpm = MODEL_TPM,
    rpd = MODEL_RPD,
    supported_parameters = { "tools", "tool_choice", "response_format", "temperature", "top_p", "max_tokens" },
    input_modalities = string_list(modalities.input, { "text" }),
    output_modalities = string_list(modalities.output, { "text" }),
    endpoints = { "chat/completions" },
  }
  if entry.reasoning == true then row.reasoning = { default_enabled = true } end
  return row
end

-- The catalog fetches are anonymous upstream (UA opencode/latest/<ver>/cli,
-- no auth or session headers), so they use bare clients, not
-- opencode_headers. Returns nil on failure (never an empty table: the router
-- rejects {} but maps nil to an empty model list).
local function refresh_model_catalog()
  local version = opencode_version()
  if version == nil then return nil end
  local headers = { ["User-Agent"] = catalog_user_agent(version), ["Accept"] = "application/json" }
  local client = llm_router.http_client({})
  local resp, err = client:request({ method = "GET", url = BASE_URL .. "/models", headers = headers })
  if err ~= nil or resp.status ~= 200 then
    llm_router.log("warn", "model list fetch failed", { status = resp and resp.status, reason = err and err.message })
    return nil
  end
  local parsed = decode_json_table(resp.body)
  if parsed == nil or type(parsed.data) ~= "table" then return nil end
  local whitelist = {}
  for _, m in ipairs(parsed.data) do
    if type(m) == "table" and is_free_model(m.id) then whitelist[m.id] = true end
  end
  local enriched = {}
  local catalog_client = llm_router.http_client({ timeout_ms = 60000 })
  local eres, eerr = catalog_client:request({ method = "GET", url = MODELS_DEV_URL, headers = headers })
  if eerr == nil and eres.status == 200 then
    local catalog = decode_json_table(eres.body)
    if catalog ~= nil and type(catalog.opencode) == "table" and type(catalog.opencode.models) == "table" then
      enriched = catalog.opencode.models
    end
  else
    llm_router.log("warn", "model enrichment fetch failed", { status = eres and eres.status, reason = eerr and eerr.message })
  end
  -- Sorted names keep the served catalog deterministic across refreshes.
  local names = {}
  for name in pairs(whitelist) do table.insert(names, name) end
  table.sort(names)
  local rows = {}
  for _, name in ipairs(names) do table.insert(rows, build_model_row(name, enriched[name])) end
  -- An empty refresh stores nothing, so the next call retries instead of
  -- pinning an empty catalog for the TTL. A failed cache write still serves
  -- the fresh rows to this caller.
  if #rows == 0 then return nil end
  store_set(MODEL_CACHE_SCOPE, "infos", rows, MODEL_CACHE_TTL)
  return rows
end

---------------------------------------------------------------------------
-- Payloads
---------------------------------------------------------------------------

-- Responses requires call_id values no longer than 64 characters. Native
-- ids stay when possible; oversized ids shorten deterministically so
-- function_call and function_call_output references remain identical.
local CALL_ID_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

local function normalize_call_id(raw)
  if #raw <= 64 then return raw end
  local normalized = "call_" .. llm_router.uuid_v5(CALL_ID_NAMESPACE, raw)
  llm_router.log("debug", "normalized call id", { from_len = #raw, to_len = #normalized })
  return normalized
end

-- Plain-text view of a chat message. Content arrives either as a string or
-- as an array of content parts (multimodal/tool messages).
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

-- Reasoning trace of an assistant message: reasoning_content, or the
-- `reasoning` spelling some upstreams use.
local function reasoning_text(m)
  local text = m.reasoning_content
  if type(text) ~= "string" or text == "" then text = m.reasoning end
  if type(text) == "string" and text ~= "" then return text end
  return nil
end

local function messages_of(request)
  if type(request.messages) == "table" then return request.messages end
  return {}
end

-- Reasoning replay. Encrypted reasoning blobs ({id, blob}) belong to the
-- assistant turn that produced them, and the history replays each one in
-- front of that turn, as the genuine client does with store=false. A turn is
-- recognized by what survives the round trip through the client: its first
-- tool call id, or else its text. Blobs live under that anchor (24-hour TTL),
-- not under a dialogue position. Positional matching handed the newest
-- response's blobs to the oldest assistant message: the replayed prefix then
-- changed on every request (no prompt-cache hit past the static head) and the
-- model saw its latest reasoning attached to its first step. A turn without
-- a stored blob replays nothing (the validator rejects reconstructed shells).
local REASONING_SCOPE = "reasoning_by_turn"
local REASONING_TTL = 24 * 3600
local ANCHOR_NS = "3b1f6c52-7a0d-4e9b-b2c8-5d4a9f0e1c73"

local function text_anchor(text)
  if text == "" then return nil end
  return "text:" .. text
end

-- Anchor of an assistant message in the client's history.
local function message_anchor(m)
  if type(m.tool_calls) == "table" then
    for _, tc in ipairs(m.tool_calls) do
      if type(tc) == "table" and is_filled_string(tc.id) then return "call:" .. tc.id end
    end
  end
  return text_anchor(message_text(m))
end

-- Keys hash the anchor: call ids and text are arbitrary strings.
local function anchor_key(anchor) return llm_router.uuid_v5(ANCHOR_NS, anchor) end

local function save_reasoning(anchor, blobs)
  if anchor == nil or #blobs == 0 then return end
  store_set(REASONING_SCOPE, anchor_key(anchor), blobs, REASONING_TTL)
end

local function load_reasoning(anchor)
  local blobs = store_get(REASONING_SCOPE, anchor_key(anchor))
  if is_filled_array(blobs) then return blobs end
  return {}
end

-- Build the OpenAI Responses input from chat messages. Returns (nil, err)
-- for conversations that cannot be expressed faithfully.
local function build_responses_input(messages)
  local input = {}
  for _, m in ipairs(messages) do
    if type(m) == "table" then
      local role = m.role
      if role == "system" then
        local text = message_text(m)
        if text ~= "" then table.insert(input, { role = "system", content = text }) end
      elseif role == "user" then
        local text = message_text(m)
        if text ~= "" then
          table.insert(input, { role = "user", content = { { type = "input_text", text = text } } })
        end
      elseif role == "assistant" then
        local anchor = message_anchor(m)
        if anchor ~= nil then
          for _, stored in ipairs(load_reasoning(anchor)) do
            if type(stored) == "table" and is_filled_string(stored.blob) then
              local item = { type = "reasoning", summary = WIRE_EMPTY_ARRAY, encrypted_content = stored.blob }
              if is_filled_string(stored.id) then item.id = stored.id end
              llm_router.log("debug", "replaying reasoning", { reasoning_id = item.id, event = "reasoning_replay" })
              table.insert(input, item)
            end
          end
        end
        local text = message_text(m)
        if text ~= "" then
          table.insert(input, { role = "assistant", content = { { type = "output_text", text = text } } })
        end
        if type(m.tool_calls) == "table" then
          for _, tc in ipairs(m.tool_calls) do
            local fn = type(tc) == "table" and tc["function"]
            if type(fn) == "table" and is_filled_string(fn.name) then
              if not is_filled_string(tc.id) then
                return nil, fault("assistant tool call without an id", "invalid_request_error", 400)
              end
              local args = fn.arguments
              if not is_filled_string(args) then args = "{}" end
              table.insert(input, { type = "function_call", call_id = normalize_call_id(tc.id), name = fn.name, arguments = args })
            end
          end
        end
      elseif role == "tool" then
        if not is_filled_string(m.tool_call_id) then
          return nil, fault("tool message without tool_call_id", "invalid_request_error", 400)
        end
        table.insert(input, { type = "function_call_output", call_id = normalize_call_id(m.tool_call_id), output = message_text(m) })
      end
    end
  end
  if #input == 0 then
    return nil, fault("request has no usable messages", "invalid_request_error", 400)
  end
  return input
end

local REASONING_EFFORTS = { minimal = true, low = true, medium = true, high = true, xhigh = true, max = true }

-- Anonymous chat payload in the genuine client's shape: the fetched agent
-- head leads every turn (no title path), temperature is never sent, tools
-- merge client definitions over the injected built-ins. `c` is the prepared
-- request context (see prepare_job).
local function build_chat_payload(c)
  local head, err = agent_prompt(c.version, c.model_name)
  if head == nil then return nil, err end
  local messages = { { role = "system", content = head } }
  for _, m in ipairs(messages_of(c.request)) do table.insert(messages, m) end
  local payload = {
    model = c.model_name,
    messages = messages,
    stream = true,
    stream_options = { include_usage = true },
    max_tokens = output_token_budget(c.request),
  }
  local top_p = tonumber(c.request.top_p)
  if top_p ~= nil and top_p > 0 then payload.top_p = top_p end
  local tools, terr = chat_tools(c.version, c.request)
  if tools == nil then return nil, terr end
  payload.tools = tools
  payload.tool_choice = c.request.tool_choice or "auto"
  if type(c.request.response_format) == "table" then payload.response_format = c.request.response_format end
  llm_router.log("debug", "built chat payload", { tools = #tools, messages = #messages, event = "payload" })
  return payload
end

-- Anonymous Responses payload. Reasoning travels only as caller effort;
-- prompt_cache_key is the raw session id, as the genuine client sends it;
-- include requests encrypted reasoning blobs.
local function build_responses_payload(c)
  local head, err = agent_prompt(c.version, c.model_name)
  if head == nil then return nil, err end
  local history, ierr = build_responses_input(messages_of(c.request))
  if history == nil then return nil, ierr end
  local input = { { role = "developer", content = head } }
  for _, item in ipairs(history) do table.insert(input, item) end
  local payload = {
    model = c.model_name,
    input = input,
    stream = true,
    max_output_tokens = output_token_budget(c.request),
    store = false,
    prompt_cache_key = c.ses,
    include = { "reasoning.encrypted_content" },
  }
  local effort = c.request.reasoning_effort
  if REASONING_EFFORTS[effort] then payload.reasoning = { effort = effort, summary = "auto" } end
  local top_p = tonumber(c.request.top_p)
  if top_p ~= nil and top_p > 0 then payload.top_p = top_p end
  local tools, terr = responses_tools(c.version, c.request)
  if tools == nil then return nil, terr end
  payload.tools = tools
  payload.tool_choice = c.request.tool_choice or "auto"
  llm_router.log("debug", "built responses payload", { tools = #tools, input = #input, event = "payload" })
  local rf = c.request.response_format
  if type(rf) == "table" then
    if rf.type == "json_object" then
      payload.text = { format = { type = "json_object" } }
    elseif rf.type == "json_schema" and type(rf.json_schema) == "table" then
      local format = { type = "json_schema", strict = true }
      if type(rf.json_schema.name) == "string" then format.name = rf.json_schema.name end
      if type(rf.json_schema.schema) == "table" then format.schema = rf.json_schema.schema end
      payload.text = { format = format }
    end
  end
  return payload
end

---------------------------------------------------------------------------
-- Upstream SSE to chat chunks
---------------------------------------------------------------------------

-- Both upstream protocols stream server-sent events. A reader turns the
-- lines of one attempt into chat.completion.chunk tables handed to `emit`:
-- the chat reader forwards upstream chunks, the Responses reader converts
-- events. Streaming forwards the chunks to the client; unary collects them
-- into one completion, so each protocol has a single translation.
-- reader.line(line) consumes one SSE line, reader.finish() runs after a
-- clean end of stream, reader.emitted tells whether anything reached `emit`.

-- Payload of an SSE data line; nil for other lines, [DONE], empty data.
local function sse_data(line)
  local data = line:match("^data: ?(.-)\r?$")
  if data == nil or data == "" or data == "[DONE]" then return nil end
  return data
end

local function log_cache_hit(usage, input_tokens)
  local details = usage.input_tokens_details
  if type(details) == "table" and tonumber(details.cached_tokens) then
    llm_router.log("debug", "cache hit", { cached_tokens = details.cached_tokens, input_tokens = input_tokens, event = "cache" })
  end
end

-- A stream that stops without its terminal event was cut off: an exit or the
-- gateway dropped the connection cleanly, which the HTTP layer cannot tell
-- from a finished stream. It must surface as a failure. Reported as a normal
-- completion it makes the agent believe its turn ended mid-work, and a tool
-- call cut off inside its arguments would reach the caller as a finished one.
-- Readers report (true) after a complete stream and (false, info) otherwise,
-- with info = {err, transient, truncated}.
local function truncated_stream()
  return {
    err = fault("upstream stream ended without a terminal event", "server_error", 502),
    transient = true,
    truncated = true,
  }
end

-- Codes that retrying elsewhere cannot fix.
local PERMANENT_FAILURE_MARKERS = { "invalid", "context_length", "content", "policy", "unsupported" }

-- Failure the upstream reported inside a 200 stream (response.failed, an
-- error event, or an error chunk on the chat protocol).
local function reported_failure(event)
  local detail = event.error
  if type(detail) ~= "table" and type(event.response) == "table" then detail = event.response.error end
  if type(detail) ~= "table" then detail = event end
  local code = tostring(detail.code or "")
  local message = "upstream reported a failure"
  if is_filled_string(detail.message) then message = "upstream stream failed: " .. detail.message:sub(1, 200) end
  local transient = true
  for _, marker in ipairs(PERMANENT_FAILURE_MARKERS) do
    if code:find(marker, 1, true) then transient = false end
  end
  local err
  if transient then
    err = fault(message, "server_error", 502)
  else
    err = fault(message, "invalid_request_error", 400)
  end
  return { err = err, transient = transient }
end

local function new_chat_reader(c, emit)
  local reader = { emitted = false }
  local terminal = false
  local failure = nil

  function reader.line(line)
    if line:match("^data: ?%[DONE%]") then
      terminal = true
      return
    end
    local data = sse_data(line)
    if data == nil then return end
    local chunk = decode_json_table(data)
    if chunk == nil then return end
    if type(chunk.error) == "table" then
      failure = failure or reported_failure(chunk)
      return
    end
    local has_choices = type(chunk.choices) == "table" and #chunk.choices > 0
    if has_choices then
      for _, choice in ipairs(chunk.choices) do
        if type(choice) == "table" and is_filled_string(choice.finish_reason) then terminal = true end
      end
    elseif chunk.usage == nil then
      return
    end
    chunk.model = c.full_model
    emit(chunk)
    reader.emitted = true
  end

  function reader.finish()
    if failure ~= nil then return false, failure end
    if not terminal then return false, truncated_stream() end
    return true
  end

  return reader
end

local function new_responses_reader(c, emit)
  local reader = { emitted = false }
  local calls = {} -- output_index -> { call_id, name, args, emitted }
  local tool_count = 0
  local first_call_id = nil -- anchor of the turn: what the client's history will carry
  local text = {}
  local usage = nil
  local finish = "stop"
  local blobs = {}
  local terminal = false
  local failure = nil

  local function send(chunk)
    chunk.model = c.full_model
    emit(chunk)
    reader.emitted = true
  end

  local function call_slot(index)
    local slot = calls[index]
    if slot == nil then
      slot = { call_id = "", name = "", args = "", emitted = false }
      calls[index] = slot
    end
    return slot
  end

  local function tool_call_entry(slot)
    local entry = {
      index = tool_count,
      id = slot.call_id,
      type = "function",
      ["function"] = { name = slot.name, arguments = slot.args },
    }
    tool_count = tool_count + 1
    slot.emitted = true
    if first_call_id == nil and slot.call_id ~= "" then first_call_id = slot.call_id end
    return entry
  end

  local function handle_function_item(event_type, item, output_index)
    local slot = call_slot(output_index)
    if is_filled_string(item.call_id) then slot.call_id = item.call_id end
    if is_filled_string(item.name) then slot.name = item.name end
    if is_filled_string(item.arguments) then slot.args = item.arguments end
    if event_type == "response.output_item.done" and slot.name ~= "" and not slot.emitted then
      send({ choices = { { index = 0, delta = { role = "assistant", tool_calls = { tool_call_entry(slot) } } } } })
    end
  end

  local function handle_terminal(event_type, response)
    terminal = true
    blobs = reasoning_blobs(response)
    if type(response.usage) == "table" then
      local prompt = response.usage.input_tokens or 0
      local completion = response.usage.output_tokens or 0
      usage = {
        prompt_tokens = prompt,
        completion_tokens = completion,
        total_tokens = response.usage.total_tokens or (prompt + completion),
      }
      log_cache_hit(response.usage, prompt)
    end
    if event_type == "response.incomplete" then
      finish = "length"
      if type(response.incomplete_details) == "table" and response.incomplete_details.reason == "content_filter" then
        finish = "content_filter"
      end
    end
  end

  function reader.line(line)
    local data = sse_data(line)
    if data == nil then return end
    local event = decode_json_table(data)
    if event == nil or type(event.type) ~= "string" then return end
    local event_type = event.type
    if event_type == "response.output_text.delta" then
      if is_filled_string(event.delta) then
        table.insert(text, event.delta)
        send({ choices = { { index = 0, delta = { role = "assistant", content = event.delta } } } })
      end
    elseif event_type == "response.reasoning_summary_text.delta" then
      if is_filled_string(event.delta) then
        send({ choices = { { index = 0, delta = { role = "assistant", reasoning_content = event.delta } } } })
      end
    elseif event_type == "response.output_item.added" or event_type == "response.output_item.done" then
      if type(event.item) == "table" and event.item.type == "function_call" then
        handle_function_item(event_type, event.item, event.output_index or 0)
      end
    elseif event_type == "response.function_call_arguments.delta" then
      if type(event.delta) == "string" then
        local slot = call_slot(event.output_index or 0)
        slot.args = slot.args .. event.delta
      end
    elseif event_type == "response.completed" or event_type == "response.incomplete" then
      if type(event.response) == "table" then handle_terminal(event_type, event.response) end
    elseif event_type == "response.failed" or event_type == "error" then
      failure = failure or reported_failure(event)
    end
  end

  -- Terminal chunk: finish reason plus usage. Tool calls emitted above stay
  -- out; anything accumulated but unemitted rides here. Only a stream that
  -- reached response.completed or response.incomplete gets one: tool calls
  -- accumulated by a cut-off stream are never delivered.
  function reader.finish()
    if failure ~= nil then return false, failure end
    if not terminal then return false, truncated_stream() end
    local indexes = {}
    for index, slot in pairs(calls) do
      if slot.name ~= "" and not slot.emitted then table.insert(indexes, index) end
    end
    table.sort(indexes)
    local pending = {}
    for _, index in ipairs(indexes) do table.insert(pending, tool_call_entry(calls[index])) end
    local final = { index = 0, delta = { role = "assistant" }, finish_reason = finish }
    if #pending > 0 then final.delta.tool_calls = pending end
    if tool_count > 0 then final.finish_reason = "tool_calls" end
    local chunk = { choices = { final } }
    if usage ~= nil then chunk.usage = usage end
    send(chunk)
    if tool_count == 0 and #text == 0 then
      llm_router.log("warn", "completion without output", { finish = finish })
    end
    local anchor = nil
    if first_call_id ~= nil then
      anchor = "call:" .. first_call_id
    else
      anchor = text_anchor(table.concat(text))
    end
    save_reasoning(anchor, blobs)
    return true
  end

  return reader
end

-- Folds the chunks of one upstream stream into a single chat.completion.
local function new_collector()
  local text, trace = {}, {}
  local calls, order = {}, {}
  local usage, finish = nil, nil

  -- Slot of a streamed tool call. Calls normally carry an index; for
  -- upstreams that omit it a new id opens a new call and anything else
  -- continues the latest one.
  local function call_slot(tc)
    local index = tc.index
    if type(index) ~= "number" then
      local last = order[#order]
      if last ~= nil and (not is_filled_string(tc.id) or tc.id == calls[last].id) then
        index = last
      else
        index = (last or -1) + 1
      end
    end
    local slot = calls[index]
    if slot == nil then
      slot = { id = "", name = "", args = {} }
      calls[index] = slot
      table.insert(order, index)
    end
    return slot
  end

  local function add_tool_call(tc)
    local slot = call_slot(tc)
    if is_filled_string(tc.id) then slot.id = tc.id end
    local fn = tc["function"]
    if type(fn) == "table" then
      if is_filled_string(fn.name) then slot.name = fn.name end
      -- Argument fragments arrive piecewise and concatenate.
      if type(fn.arguments) == "string" then table.insert(slot.args, fn.arguments) end
    end
  end

  local function add_choice(choice)
    local delta = choice.delta
    if type(delta) == "table" then
      if is_filled_string(delta.content) then table.insert(text, delta.content) end
      local reasoning = reasoning_text(delta)
      if reasoning ~= nil then table.insert(trace, reasoning) end
      if type(delta.tool_calls) == "table" then
        for _, tc in ipairs(delta.tool_calls) do
          if type(tc) == "table" then add_tool_call(tc) end
        end
      end
    end
    if is_filled_string(choice.finish_reason) then finish = choice.finish_reason end
  end

  local collector = {}

  function collector.add(chunk)
    if type(chunk.choices) == "table" then
      for _, choice in ipairs(chunk.choices) do
        if type(choice) == "table" then add_choice(choice) end
      end
    end
    if type(chunk.usage) == "table" then
      local prompt = chunk.usage.prompt_tokens or 0
      local completion = chunk.usage.completion_tokens or 0
      usage = {
        prompt_tokens = prompt,
        completion_tokens = completion,
        total_tokens = chunk.usage.total_tokens or (prompt + completion),
      }
    end
  end

  function collector.build(model)
    table.sort(order)
    local tool_calls = {}
    for _, index in ipairs(order) do
      local slot = calls[index]
      if slot.name ~= "" then
        table.insert(tool_calls, {
          id = slot.id,
          type = "function",
          ["function"] = { name = slot.name, arguments = table.concat(slot.args) },
        })
      end
    end
    local finish_reason = finish or "stop"
    if #tool_calls > 0 and finish_reason == "stop" then finish_reason = "tool_calls" end
    if #tool_calls == 0 then tool_calls = empty_array() end
    local content = table.concat(text)
    local message = { role = "assistant", content = content, tool_calls = tool_calls }
    local reasoning = table.concat(trace)
    if reasoning ~= "" then message.reasoning_content = reasoning end
    usage = usage or { prompt_tokens = 0, completion_tokens = 0, total_tokens = 0 }
    llm_router.log("debug", "assembled completion", {
      finish = finish_reason,
      prompt_tokens = usage.prompt_tokens,
      completion_tokens = usage.completion_tokens,
      text_chars = #content,
    })
    return {
      id = "chatcmpl-" .. llm_router.random_hex(12),
      object = "chat.completion",
      created = os.time(),
      model = model,
      choices = { { index = 0, message = message, finish_reason = finish_reason } },
      usage = usage,
    }
  end

  return collector
end

---------------------------------------------------------------------------
-- Requests
---------------------------------------------------------------------------

-- Protocol families first: muse-spark/gpt/grok serve /responses even for
-- their -free variants. Remaining -free models are chat-protocol.
local PROTOCOLS = {
  chat = { path = "chat/completions", build_payload = build_chat_payload, new_reader = new_chat_reader },
  responses = { path = "responses", build_payload = build_responses_payload, new_reader = new_responses_reader },
}

local function protocol_for_model(model_name)
  local m = model_name:lower()
  if m:match("^gpt%-") or m:match("muse%-spark") or m:match("^grok%-") then return PROTOCOLS.responses end
  return PROTOCOLS.chat
end

-- Everything that does not depend on the exit: session, payload, wire body,
-- headers. Built once per request, before any proxy is requested, so a
-- malformed request or missing remote text fails at once. Returns (nil, err)
-- on failure.
local function prepare_job(request)
  local model_name = request.model_name
  local protocol = protocol_for_model(model_name)
  local version, verr = opencode_version()
  if version == nil then return nil, verr end
  local keys, kerr = dialogue_keys(request, model_name)
  if keys == nil then return nil, kerr end
  local job = {
    request = request,
    model_name = model_name,
    full_model = request.model,
    protocol = protocol,
    version = version,
    keys = keys,
    ses = resolve_session(keys),
    url = BASE_URL .. "/" .. protocol.path,
  }
  local payload, perr = protocol.build_payload(job)
  if payload == nil then return nil, perr end
  local body, eerr = wire_body(payload)
  if body == nil then return nil, eerr end
  job.body = body
  job.headers = opencode_headers(version, job.ses, mint_msg(mint_ms()))
  return job
end

local function log_request_start(label, request, job_protocol)
  llm_router.log("debug", label, {
    model = request.model,
    endpoint = job_protocol.path,
    tools = #client_tools(request.tools),
    messages = #messages_of(request),
    event = "request_start",
  })
end

-- Whether a failed attempt may move to another exit: the client marks
-- transport faults that happened on a proxied leg before any stream output
-- reached the caller as retryable. Everything else (cancellation, direct
-- leg, faults after output started) is final.
local function transport_verdict(err)
  if err.retryable == true then return NEXT end
  return STOP
end

-- Settles a reader after a clean end of stream. A cut-off or failed stream
-- never counts as a completion: before anything reached the caller the
-- request moves to the next exit (when the failure looks transient), after
-- it the failure is final. `streaming` is false for unary requests, whose
-- chunks only feed the collector and have not reached the caller.
local function close_reader(px, reader, streaming)
  local ok, info = reader.finish()
  if ok then return "ok" end
  if info.truncated then mark_exit(px, "server_error", SERVER_ERROR_TTL) end
  if (streaming and reader.emitted) or not info.transient then return STOP, info.err end
  return NEXT, info.err
end

-- One unary attempt. Returns (body) or (nil, verdict, err).
local function attempt_unary(run, client, job, px)
  local resp, err = client:request({
    method = "POST",
    url = job.url,
    headers = job.headers,
    body = job.body,
    proxy_url = px.url,
  })
  if err ~= nil then
    mark_exit_transport(px, err)
    return nil, transport_verdict(err), err
  end
  if resp.status == 200 then return resp.body end
  dump_request(job)
  local verdict, mapped = map_upstream(run, resp, job, px)
  return nil, verdict, mapped
end

-- One streaming attempt. Returns "ok", or (verdict, err).
local function attempt_stream(run, client, job, px, emit)
  local reader = job.protocol.new_reader(job, emit)
  local mapped_verdict = nil
  local _, err = client:stream({
    method = "POST",
    url = job.url,
    headers = job.headers,
    body = job.body,
    proxy_url = px.url,
    on_response = function(head)
      llm_router.log("debug", "stream responded", { status = head.status, event = "stream_response" })
      if head.status ~= 200 then
        dump_request(job)
        local verdict, mapped = map_upstream(run, head, job, px)
        mapped_verdict = verdict
        return mapped
      end
    end,
    on_line = function(line)
      llm_router.log("debug", "server-sent event", { line = line, event = "sse" })
      reader.line(line)
    end,
  })
  if err == nil then return close_reader(px, reader, true) end
  mark_exit_transport(px, err)
  -- The stream belongs to the attempt once output started: no failover.
  if reader.emitted then return STOP, err end
  return mapped_verdict or transport_verdict(err), err
end

local function log_attempt_failure(run, event, message, verdict, err, attempt)
  -- Rate limits and denials have their own counters; anything else is the
  -- real cause the terminal message should not hide.
  if verdict == NEXT and err.status ~= 429 and err.status ~= 403 then run.last_failure = err.message end
  local severity = "warn"
  if verdict == STOP then severity = "error" end
  llm_router.log(severity, message, {
    code = err.code,
    status = err.status,
    reason = err.message,
    attempt = attempt,
    event = event,
  })
end

-- Runs `attempt(px, attempt_number)` once per exit until it reports a final
-- outcome: it returns (false) to try the next exit, or (true, value, err) to
-- finish with that result. Exhausting the exits ends in the terminal 429. A
-- last-resort retry that succeeds lifts its mark: a pair park clears the pair,
-- an exit-wide block clears the exit.
local function run_attempts(ctx, run, job, attempt)
  local next_exit, exits = exit_stream(ctx.provider_config, job)
  local attempted = 0
  while true do
    local px, kind = next_exit()
    if px == nil then break end
    attempted = attempted + 1
    local final, value, err = attempt(px, attempted)
    if kind ~= nil and final and err == nil then
      if kind == "parked" then clear_pair(px, job.model_name) else clear_exit(px) end
    end
    if final then return value, err end
  end
  return nil, terminal_error(run, exits, attempted)
end

---------------------------------------------------------------------------
-- Provider registration
---------------------------------------------------------------------------

llm_router.register("opencode-free", {
  icon = "https://opencode.ai/favicon.ico",

  proxy_schema = {},

  get_model_infos = function(ctx)
    local cached = store_get(MODEL_CACHE_SCOPE, "infos")
    if is_filled_array(cached) then
      llm_router.log("debug", "model catalog loaded", { count = #cached })
      return cached
    end
    llm_router.log("debug", "model catalog missed")
    return refresh_model_catalog()
  end,

  -- Model liveness: the tier is anonymous, so a single direct
  -- discriminator POST proves existence with zero generation spend
  -- (rejection happens pre-generation). Free names answer 403
  -- FreeTierError (gated to genuine clients, not missing); paid names answer
  -- 401 (not on tier); unknown names answer 400 Model-unavailable. Probes
  -- never write storage marks.
  check_model = function(ctx, model_ref)
    local capability = model_ref.endpoint or "chat"
    if capability ~= "chat" and capability ~= "vision" then
      return { status = "unknown", message = "endpoint not supported: " .. tostring(capability) }
    end
    local model_name = model_ref.model_name or ""
    if model_name == "" then return { status = "unknown", message = "model name is required" } end
    local version = opencode_version()
    if version == nil then return { status = "unknown", message = "client version unavailable" } end
    local client = llm_router.http_client({ timeout_ms = 15000 })
    local ms = mint_ms()
    llm_router.log("debug", "probing model", { model = model_ref.model, endpoint = capability })
    local resp, err = client:request({
      method = "POST",
      url = BASE_URL .. "/chat/completions",
      headers = opencode_headers(version, mint_session(ms), mint_msg(ms)),
      body = json.encode({ model = model_name, messages = { { role = "user", content = "hi" } }, max_tokens = 1 }),
    })
    if err ~= nil then
      return { status = "unknown", message = type(err.message) == "string" and err.message or "request failed" }
    end
    local status = resp.status
    local body = tostring(resp.body or "")
    if status == 200 then return { status = "healthy", response = "ok" } end
    if status == 400 and body:find("Model is unavailable", 1, true) then
      return { status = "unhealthy", message = "model not found upstream" }
    end
    if status == 401 and not is_free_model(model_name) then
      return { status = "unhealthy", message = "model is not available on the free tier" }
    end
    if status == 403 and body:find(FINGERPRINT_MARKER, 1, true) then
      return { status = "healthy", response = "ok" }
    end
    if status == 429 then
      if is_quota_body(body) then return { status = "quota_exceeded", message = "quota exhausted" } end
      return { status = "unknown", message = "rate limited" }
    end
    return { status = "unknown", message = "status " .. tostring(status) }
  end,

  complete = function(ctx, request)
    log_request_start("starting request", request, protocol_for_model(request.model_name))
    local job, err = prepare_job(request)
    if job == nil then return nil, err end
    local run = new_run()
    local client = llm_router.http_client({ timeout_ms = GENERATION_TIMEOUT_MS })
    return run_attempts(ctx, run, job, function(px, attempt)
      llm_router.log("debug", "sending request", { endpoint = job.protocol.path, body_len = #job.body, attempt = attempt, event = "http" })
      local body, verdict, ferr = attempt_unary(run, client, job, px)
      if body ~= nil then
        local collector = new_collector()
        local reader = job.protocol.new_reader(job, collector.add)
        for line in (body .. "\n"):gmatch("([^\n]*)\n") do reader.line(line) end
        local outcome, rerr = close_reader(px, reader, false)
        if outcome == "ok" then return true, collector.build(job.full_model) end
        verdict, ferr = outcome, rerr
      end
      log_attempt_failure(run, "http_error", "request failed", verdict, ferr, attempt)
      return verdict == STOP, nil, ferr
    end)
  end,

  complete_stream = function(ctx, request, emit)
    log_request_start("starting stream", request, protocol_for_model(request.model_name))
    local job, err = prepare_job(request)
    if job == nil then return nil, err end
    local run = new_run()
    local client = llm_router.http_client({ timeout_ms = GENERATION_TIMEOUT_MS })
    return run_attempts(ctx, run, job, function(px, attempt)
      llm_router.log("debug", "sending stream", { endpoint = job.protocol.path, body_len = #job.body, attempt = attempt, event = "stream" })
      local outcome, serr = attempt_stream(run, client, job, px, emit)
      if outcome == "ok" then
        llm_router.log("debug", "stream completed", { endpoint = job.protocol.path, event = "stream_done" })
        return true
      end
      log_attempt_failure(run, "stream_error", "stream failed", outcome, serr, attempt)
      return outcome == STOP, nil, serr
    end)
  end,

  -- Verified reasoning support the upstream catalog does not advertise.
  -- Effort sets match the opencode-go registry measurements.
  model_specs = {
    ["mimo-v2.6-flash-free"] = {
      reasoning = { supported_efforts = { "high", "max" } },
      input_modalities = { "text", "image" },
      output_modalities = { "text" },
    },
    ["muse-spark-1.2-contributor-free"] = {
      reasoning = { supported_efforts = { "minimal", "low", "medium", "high", "xhigh" } },
    },
    ["muse-spark-1.3-contributor-free"] = {
      reasoning = { supported_efforts = { "minimal", "low", "medium", "high", "xhigh" } },
    },
  },
})
