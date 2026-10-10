--- @plugin OpenCode Free
--- @author TheSlopMachine
--- @version 6.3.2
--- @plugin_api 1.0
--- @description OpenAI/Anthropic/Google compatible free provider OpenCode Free (no key required)
--- @allow_host opencode.ai
--- @allow_host models.opencode.ai
--- @allow_host registry.npmjs.org
--- @allow_host raw.githubusercontent.com

local BASE_URL = "https://opencode.ai/zen/v1"
local MODELS_DEV_URL = "https://models.opencode.ai/api.json"
local NPM_VERSION_URL = "https://registry.npmjs.org/opencode-ai/latest"
local RAW_BASE = "https://raw.githubusercontent.com/anomalyco/opencode/"
local RAW_BRANCHES = { "dev", "main" }

-- Protocol families first: muse-spark/gpt/grok serve /responses even for
-- their -free variants. Remaining -free models are chat-protocol.
local function endpoint_for_model(model)
  local m = model:lower()
  if m:match("^gpt%-") or m:match("muse%-spark") or m:match("^grok%-") then return "/responses" end
  return "/chat/completions"
end

-- Free-tier models think inside the output budget, including chat-protocol
-- models with no reasoning flag. Requests below the floor starve the text
-- (reasoning consumes the whole budget), so the floor applies to every
-- model, explicit caller budgets included.
local MIN_OUTPUT_TOKENS = 512

local function debug_json(value)
  local ok, encoded = pcall(json.encode, value)
  if ok and type(encoded) == "string" then return encoded end
  return "<json.encode failed: " .. tostring(encoded) .. ">"
end

local function debug_log(tag, message)
  local ok = pcall(print, "[opencode-free][" .. tostring(tag) .. "] " .. tostring(message))
  if not ok then
  end
end

local function table_shape(value)
  if type(value) ~= "table" then return type(value) end
  local numeric = 0
  local other = 0
  local max_index = 0
  for k, _ in pairs(value) do
    if type(k) == "number" and k >= 1 and k % 1 == 0 then
      numeric = numeric + 1
      if k > max_index then max_index = k end
    else
      other = other + 1
    end
  end
  if other == 0 and numeric == 0 then return "table(empty)" end
  if other == 0 and max_index == numeric then return "array(" .. tostring(numeric) .. ")" end
  return "map(numeric=" .. tostring(numeric) .. ",other=" .. tostring(other) .. ")"
end

local function tool_name(t)
  if type(t) ~= "table" then return "" end
  if type(t["function"]) == "table" and type(t["function"].name) == "string" then return t["function"].name end
  if type(t.name) == "string" then return t.name end
  return ""
end

local function normalize_tools(tools)
  local out = {}
  if type(tools) ~= "table" then return out end

  local function add_tool(t)
    if type(t) == "table" then table.insert(out, t) end
  end

  local has_array_items = false
  for i = 1, #tools do
    has_array_items = true
    add_tool(tools[i])
  end
  if has_array_items then return out end

  local looks_like_single_tool = type(tools.name) == "string"
    or (type(tools["function"]) == "table" and type(tools["function"].name) == "string")
    or type(tools.input_schema) == "table"
  if looks_like_single_tool then
    add_tool(tools)
    return out
  end

  -- Map-shaped input iterates in undefined order. Sort by name so the
  -- upstream tool list is stable.
  for _, value in pairs(tools) do
    add_tool(value)
  end
  table.sort(out, function(a, b) return tool_name(a) < tool_name(b) end)
  return out
end

local function output_token_budget(request)
  local value = tonumber(request.max_completion_tokens)
  if value == nil or value <= 0 then value = tonumber(request.max_tokens) end
  if value == nil or value <= 0 then return 32000 end
  if value < MIN_OUTPUT_TOKENS then return MIN_OUTPUT_TOKENS end
  return value
end

-- Remote text cache: every fingerprint-critical byte (CLI version, prompt
-- heads, tool descriptions) arrives over the wire and refreshes on TTL.
-- Fetch failures serve last-cached bytes; prompt callers fail closed when
-- nothing is cached, since a wrong fingerprint is worse than an error.
local TEXT_CACHE_SCOPE = "remote_text"

local function fetch_text_cached(url, cache_key, ttl)
  local ok, cached = pcall(llm_router.storage.get, TEXT_CACHE_SCOPE, cache_key)
  if ok and type(cached) == "string" and cached ~= "" then return cached end
  local client = llm_router.http_client({ timeout_ms = 30000 })
  local resp, err = client:request({ method = "GET", url = url, headers = {} })
  if err or resp == nil or resp.status ~= 200 or type(resp.body) ~= "string" or resp.body == "" then
    return nil
  end
  pcall(llm_router.storage.set, TEXT_CACHE_SCOPE, cache_key, resp.body, { ttl = ttl })
  return resp.body
end

-- CLI version for the User-Agent, from the npm registry metadata the
-- official upgrade path prefers over the rate-limited GitHub API.
-- Verified live: /latest answers {"version": "1.18.35", ...} with no auth.
local VERSION_TTL = 86400
local VERSION_FALLBACK = "1.18.35"

local function opencode_version()
  local body = fetch_text_cached(NPM_VERSION_URL, "npm_version", VERSION_TTL)
  if type(body) == "string" and body ~= "" then
    local ok, parsed = pcall(json.decode, body)
    if ok and type(parsed) == "table" and type(parsed.version) == "string" and parsed.version ~= "" then
      return parsed.version
    end
  end
  return VERSION_FALLBACK
end

-- provider-utils sub-version has no live source; the observed constants are
-- keyed by endpoint family (chat 4.0.23, responses 4.0.40 on 1.18.35).
local function user_agent(endpoint)
  local sdk = "4.0.23"
  if endpoint == "/responses" then sdk = "4.0.40" end
  return "opencode/" .. opencode_version() .. " ai-sdk/provider-utils/" .. sdk .. " runtime/bun/1.3.14"
end

local function catalog_ua() return "opencode/latest/" .. opencode_version() .. "/cli" end

-- Prompt heads fetched from the opencode repository (dev branch first, main
-- as fallback). SystemPrompt.provider owns the family mapping; the same
-- text serves both protocols, only the role wrapper differs.
local PROMPT_TTL = 86400
local PROMPT_DIR = "packages/opencode/src/session/prompt/"

local function fetch_repo_text(path, cache_key)
  local ok, cached = pcall(llm_router.storage.get, TEXT_CACHE_SCOPE, cache_key)
  if ok and type(cached) == "string" and cached ~= "" then return cached end
  for _, branch in ipairs(RAW_BRANCHES) do
    local body = fetch_text_cached(RAW_BASE .. branch .. "/" .. path, cache_key .. ":" .. branch, PROMPT_TTL)
    if type(body) == "string" and body ~= "" then
      pcall(llm_router.storage.set, TEXT_CACHE_SCOPE, cache_key, body, { ttl = PROMPT_TTL })
      return body
    end
  end
  if ok and type(cached) == "string" then return cached end
  return nil
end

local function prompt_file_for_model(model)
  local m = model:lower()
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
-- Returns (nil, err) when neither fresh nor cached bytes exist.
local function agent_prompt(model)
  local file, is_meta = prompt_file_for_model(model)
  local head = fetch_repo_text(PROMPT_DIR .. file, "prompt:" .. file)
  if type(head) ~= "string" or head == "" then
    return nil, { message = "agent prompt unavailable (" .. file .. ")", code = "server_error", status = 502 }
  end
  if is_meta then
    local name = "Muse Spark"
    if model:lower():find("muse-glimmer", 1, true) then name = "Muse Glimmer" end
    head = head:gsub("{{MODEL_NAME}}", name)
  end
  return head
end

-- Tool description prose fetched from the tool sibling .txt files.
-- Parameter schemas below are structural (TS-annotation-sourced) and pinned
-- against live captures, not prose.
local TOOL_DESC_DIR = "packages/opencode/src/tool/"
local TOOL_DESC_FILES = {
  read = "read.txt",
  write = "write.txt",
  edit = "edit.txt",
  glob = "glob.txt",
  grep = "grep.txt",
}

local function tool_description(name)
  local file = TOOL_DESC_FILES[name]
  if file == nil then return nil end
  return fetch_repo_text(TOOL_DESC_DIR .. file, "tooldesc:" .. name)
end

-- pwsh/win32 render of tool/shell/shell.txt, ported from
-- tool/shell/prompt.ts (powershellCommandSection + chainGuidance PS branch,
-- limits 2000 lines / 51200 bytes, 120000ms default timeout).
-- The tmp path is machine-specific upstream; the fallback preserves the
-- proven Windows-client bytes and is documented, not derived.
local SHELL_TMP_FALLBACK = "C:\\Users\\Thinker\\AppData\\Local\\Temp\\opencode"

local function shell_description()
  local template = fetch_repo_text(TOOL_DESC_DIR .. "shell/shell.txt", "tooldesc:shell-template")
  if type(template) ~= "string" or template == "" then return nil end
  local chain = "If the commands depend on each other and must run sequentially, use a single bash tool call with '&&' to chain them together (e.g., `git add . && git commit -m \"message\" && git push`). For instance, if one operation must complete before another starts (like New-Item before Copy-Item, Write before bash for git operations, or git add before git commit), run these operations sequentially instead."
  local notes = "# PowerShell (7+) shell notes\n"
    .. "- This cross-platform shell supports pipeline chain operators (`&&` and `||`).\n"
    .. "- Use double quotes for interpolated strings (`\"Hello $name\"`), single quotes for verbatim strings.\n"
    .. "- Prefer full cmdlet names like `Get-ChildItem`, `Set-Content`, `Remove-Item`, and `New-Item` over aliases.\n"
    .. "- Use `$(...)` for subexpressions. Use `@(...)` for array expressions.\n"
    .. "- To call a native executable whose path contains spaces, use the call operator: `& \"path/to/exe\" args`.\n"
    .. "- Escape special characters with the PowerShell backtick character."
  local command_section = notes
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
    .. chain
    .. "\n     - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail\n"
    .. "     - DO NOT use newlines to separate commands (newlines are ok in quoted strings)\n"
    .. "   - AVOID changing directories inside the command. Use the `workdir` parameter to change directories instead.\n"
    .. "     <good-example>\n"
    .. "     Use workdir=\"project\\subdir\" with command: pytest tests\n"
    .. "     </good-example>\n"
    .. "     <bad-example>\n"
    .. "     Set-Location -LiteralPath \"project\\subdir\" && pytest tests\n"
    .. "     </bad-example>"
  local workdir_section = "All commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID changing directories inside the command - use `workdir` instead."
  local rendered = template
  rendered = rendered:gsub(
    "${intro}",
    "Executes a given PowerShell (7+) command with optional timeout, ensuring proper handling and security measures.",
    1
  )
  rendered = rendered:gsub("${os}", "win32", 1)
  rendered = rendered:gsub("${shell}", "pwsh", 1)
  rendered = rendered:gsub("${tmp}", SHELL_TMP_FALLBACK, 1)
  rendered = rendered:gsub("${workdirSection}", workdir_section, 1)
  rendered = rendered:gsub("${commandSection}", command_section, 1)
  return rendered
end

local SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
local HUGE_INT = 9007199254740991

local function tool_params(name)
  if name == "bash" then
    return {
      ["$schema"] = SCHEMA_DRAFT,
      type = "object",
      properties = {
        command = { type = "string", description = "The command to execute" },
        timeout = { minimum = -HUGE_INT, exclusiveMinimum = 0, type = "integer", maximum = HUGE_INT },
        workdir = { type = "string" },
      },
      required = { "command" },
    }
  elseif name == "read" then
    return {
      ["$schema"] = SCHEMA_DRAFT,
      type = "object",
      properties = {
        filePath = { type = "string", description = "The absolute path to the file or directory to read" },
        offset = { minimum = 0, type = "integer", maximum = HUGE_INT },
        limit = { minimum = 0, type = "integer", maximum = HUGE_INT },
      },
      required = { "filePath" },
    }
  elseif name == "write" then
    return {
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
    }
  elseif name == "edit" then
    return {
      ["$schema"] = SCHEMA_DRAFT,
      type = "object",
      properties = {
        filePath = { type = "string", description = "The absolute path to the file to modify" },
        oldString = { type = "string", description = "The text to replace" },
        newString = { type = "string", description = "The text to replace it with (must be different from oldString)" },
        replaceAll = { type = "boolean", description = "Replace all occurrences of oldString (default false)" },
      },
      required = { "filePath", "oldString", "newString" },
    }
  elseif name == "glob" then
    return {
      ["$schema"] = SCHEMA_DRAFT,
      type = "object",
      properties = {
        pattern = { type = "string", description = "The glob pattern to match files against" },
        path = { type = "string", description = "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided." },
      },
      required = { "pattern" },
    }
  elseif name == "grep" then
    return {
      ["$schema"] = SCHEMA_DRAFT,
      type = "object",
      properties = {
        pattern = { type = "string", description = "The regex pattern to search for in file contents" },
        path = { type = "string", description = "The directory to search in. Defaults to the current working directory." },
        include = { type = "string", description = 'File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")' },
      },
      required = { "pattern" },
    }
  end
  return nil
end

local INJECTED_TOOL_NAMES = { "bash", "read", "write", "edit", "glob", "grep" }

-- Genuine main turns always carry tools. When the caller sends none, inject
-- the six built-ins (descriptions fetched, schemas structural). Returns
-- (nil, err) when a description is unavailable.
local function injected_tools()
  local out = {}
  for _, name in ipairs(INJECTED_TOOL_NAMES) do
    local desc
    if name == "bash" then
      desc = shell_description()
    else
      desc = tool_description(name)
    end
    if type(desc) ~= "string" or desc == "" then
      return nil, { message = "tool description unavailable (" .. name .. ")", code = "server_error", status = 502 }
    end
    table.insert(out, {
      type = "function",
      ["function"] = { name = name, description = desc, parameters = tool_params(name) },
    })
  end
  return out
end

local function merge_tools(client_tools)
  local out = json.decode("[]")
  local seen = {}
  for _, t in ipairs(normalize_tools(client_tools)) do
    local name = tool_name(t)
    if name ~= "" then seen[name] = true end
    table.insert(out, t)
  end
  local injected, err = injected_tools()
  if injected == nil then return nil, err end
  for _, t in ipairs(injected) do
    local name = tool_name(t)
    if name == "" or not seen[name] then table.insert(out, t) end
  end
  return out
end

local function merge_tools_resp(client_tools)
  local out = json.decode("[]")
  local seen = {}
  for _, t in ipairs(client_tools or {}) do
    local name = ""
    if type(t) == "table" then
      if type(t["function"]) == "table" and type(t["function"].name) == "string" then
        name = t["function"].name
      elseif type(t.name) == "string" then
        name = t.name
      end
    end
    if name ~= "" then seen[name] = true end
    if type(t) == "table" and type(t.name) == "string" then
      table.insert(out, t)
    else
      table.insert(out, {
        type = "function",
        name = name,
        description = (type(t["function"]) == "table" and t["function"].description) or "",
        parameters = (type(t["function"]) == "table" and t["function"].parameters)
          or { type = "object", properties = {}, required = json.decode("[]") },
      })
    end
  end
  local injected, err = injected_tools()
  if injected == nil then return nil, err end
  for _, t in ipairs(injected) do
    local fn = t["function"] or {}
    local resp_tool = {
      type = "function",
      name = fn.name or "",
      description = fn.description or "",
      parameters = fn.parameters or { type = "object", properties = {} },
    }
    if resp_tool.name == "" or not seen[resp_tool.name] then table.insert(out, resp_tool) end
  end
  return out
end

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
    error("opencode-free: llm_router.random_hex(" .. tostring(n) .. ") failed")
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

-- ULID-shaped ids. The time head uses the current second plus a uniform
-- sub-second part, matching the genuine millisecond distribution.
local function mint_ms()
  local jitter = tonumber(rand_bytes(2):sub(1, 4), 16) or 0
  return os.time() * 1000 + (jitter % 1000)
end

local function mint_session(ms) return "ses_" .. time_head(true, ms, 1) .. ulid_tail() end

local function mint_msg(ms) return "msg_" .. time_head(false, ms, 2) .. ulid_tail() end

-- Session id stable per conversation: keyed by the router's stable
-- cross-turn partition, minted once per key, 10-minute TTL so abandoned
-- sessions expire. The same id serves as prompt_cache_key verbatim.
local SESSION_SCOPE = "session_v2"
local SESSION_TTL = 600

local function conv_key(request, model)
  if type(request.cache_key) == "string" and request.cache_key ~= "" then
    return model .. "|" .. request.cache_key
  end
  local first = ""
  if type(request.messages) == "table" and type(request.messages[1]) == "table" then
    first = debug_json(request.messages[1]):sub(1, 256)
  end
  return model .. "|" .. first
end

local function stable_session(request, model)
  local key = conv_key(request, model)
  local ok, ses = pcall(llm_router.storage.get, SESSION_SCOPE, key)
  if ok and type(ses) == "string" and ses ~= "" then
    pcall(llm_router.storage.set, SESSION_SCOPE, key, ses, { ttl = SESSION_TTL })
    return ses
  end
  local fresh = mint_session(mint_ms())
  pcall(llm_router.storage.set, SESSION_SCOPE, key, fresh, { ttl = SESSION_TTL })
  return fresh
end

-- Free-tier identity: -free suffix or the known keyless set. Unknown
-- non-suffixed ids default to premium (fail-safe). Proven-broken free
-- ids stay listed nowhere: remove them here when upstream heals.
-- 2026-09-30: jev-1.13-free (upstream 502s), deepseek-v4-flash-free and
-- ling-3.0-flash-fin-free (fatal 400s).
local KNOWN_FREE = { ["big-pickle"] = true }
local EXCLUDED_FREE = {
  ["jev-1.13-free"] = true,
  ["deepseek-v4-flash-free"] = true,
  ["ling-3.0-flash-fin-free"] = true,
}

local function is_free_model(id)
  if type(id) ~= "string" or id == "" then return false end
  if EXCLUDED_FREE[id] then return false end
  if id:sub(-5) == "-free" then return true end
  return KNOWN_FREE[id] == true
end

-- Free tier is anonymous: every request carries Bearer public, the
-- ULID-shaped session pair, and the run-scoped request id.
local function opencode_headers(endpoint, ses, msg)
  return {
    ["User-Agent"] = user_agent(endpoint),
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

-- Exit-IP rate-limit memory. The anonymous tier enforces its quota per exit
-- IP and the limit lifts after about six hours. A 429 marks the exit in
-- plugin storage with that TTL; selection skips marked exits, so a limited
-- IP is never retried until the entry expires. Expired rows read as missing.
-- Storage faults fail open: an unmarked exit only costs one extra attempt.
local RL_SCOPE = "exit_limited"
local RL_TTL = 6 * 3600
local RL_NS = "6d3c2b0e-8f55-4c0a-9d4e-1b7a5e2f9c31"
local MAX_PROXY_CANDIDATES = 30
-- Exits tried per request. 403 denials are common on shared egress and fail
-- fast, so the budget leaves room for several of them.
local EXIT_ATTEMPTS = 5
local MAX_PICK_ROUNDS = 4
-- A 403 marks the exit for an hour: blocked egress addresses recover sooner
-- than rate-limited ones.
local FORBIDDEN_TTL = 3600
-- Country filter for pooled exits. Empty accepts any country.
local PROXY_COUNTRIES = {}
-- Exits that answered 403 during this request (the plugin state is fresh per
-- request), reported as one aggregated error when every attempt is denied.
local denied = 0

-- Keys hash the proxy URL so embedded credentials never land in storage.
-- The empty URL (direct connection) maps to its own fixed key.
local function exit_key(px)
  local url = (type(px) == "table" and px.url) or ""
  if url == "" then return "direct" end
  local ok, id = pcall(llm_router.uuid_v5, RL_NS, url)
  if ok and type(id) == "string" and id ~= "" then return id end
  return url
end

local function exit_limited(px)
  local ok, v = pcall(llm_router.storage.get, RL_SCOPE, exit_key(px))
  return ok and v ~= nil
end

local function mark_exit_limited(px, reason, ttl)
  pcall(llm_router.storage.set, RL_SCOPE, exit_key(px), { at = os.time(), reason = reason }, { ttl = ttl or RL_TTL })
end

-- Returns (exits, skipped): up to `limit` exits not currently marked, and
-- how many marked exits were bypassed. The pool searches for live exits and
-- waits for them; the wait deadline is shared by every round of the request.
-- A direct pool or an expired wait degrades to one direct attempt, and the
-- direct leg obeys the same mark.
local function pick_proxies(ctx, limit)
  limit = limit or 3
  -- Unconfigured providers go direct: only an explicit pool selection
  -- (dashboard proxy switch) routes through pooled exits.
  local pool = nil
  if ctx.provider_config and ctx.provider_config.proxy and ctx.provider_config.proxy.pool ~= "" then
    pool = ctx.provider_config.proxy.pool
  end
  local usable, skipped = {}, 0
  local seen = {}
  for _ = 1, MAX_PICK_ROUNDS do
    -- Ask wider than the attempt budget so marked exits at the head of the
    -- pool do not starve the fresh ones behind them. Once something usable
    -- exists, later rounds never wait for more.
    local opts = {
      pool = pool,
      countries = PROXY_COUNTRIES,
      exclude = seen,
      limit = MAX_PROXY_CANDIDATES,
      fallback = "direct",
    }
    if #usable > 0 then opts.timeout_ms = 1 end
    local res = llm_router.proxies.require(opts)
    if not res then break end
    if #res.proxies == 0 then
      -- Direct pool or wait deadline: the direct exit, unless exits are in hand.
      if #usable == 0 then
        local direct = {}
        if exit_limited(direct) then skipped = skipped + 1 else table.insert(usable, direct) end
      end
      break
    end
    for _, px in ipairs(res.proxies) do
      table.insert(seen, px.url)
      if exit_limited(px) then
        skipped = skipped + 1
      elseif #usable < limit then
        table.insert(usable, px)
      end
    end
    if #usable >= limit then break end
  end
  if skipped > 0 then
    debug_log("EXITS", "usable=" .. tostring(#usable) .. " skipped_rate_limited=" .. tostring(skipped))
  end
  return usable, skipped
end

-- Terminal error when the loop ends without an upstream error to report.
local function exits_exhausted(skipped)
  if skipped > 0 then
    return {
      message = "free-tier quota exhausted on every available exit (limited exits are skipped for 6 hours)",
      code = "insufficient_quota",
      status = 429,
    }
  end
  return { message = "all free-tier exits exhausted", code = "server_error" }
end

-- Final error of an exhausted attempt loop. When every attempted exit was
-- denied with 403 the client gets one aggregated error, never the answer of
-- a single arbitrary exit.
local function exhausted_error(last_err, skipped)
  if denied > 0 and (last_err == nil or last_err.status == 403) then
    return {
      message = "free tier denied every attempted exit with HTTP 403 (" .. tostring(denied) .. " tried)",
      code = "server_error",
      status = 502,
    }
  end
  return last_err or exits_exhausted(skipped)
end

-- Free-tier mapping. The tier is anonymous and every 429 binds to the exit
-- IP, quota wording included (the quota limiter answers per IP). The exit
-- is marked for six hours and the request moves to the next one; when all
-- exits are limited the last 429 reaches the client. Outcomes: "proxy" tries
-- the next exit, "done" returns the terminal error at once.
local function map_upstream(resp, model_name, px)
  local body_str = tostring(resp.body or "")
  if resp.status == 401 and body_str:find("only be used from within OpenCode", 1, true) then
    return "done", { message = "free tier rejected the client fingerprint", code = "server_error", status = 401 }
  end
  -- Paid models answer 401 without an API key binding on the anonymous
  -- tier: fail the request, never mark anything.
  if resp.status == 401 and not is_free_model(model_name) then
    return "done",
      {
        message = "model '" .. tostring(model_name) .. "' is not available on the free tier",
        code = "invalid_request_error",
        status = 401,
      }
  end
  if resp.status == 401 then
    return "done",
      { message = "free tier rejected the anonymous credential", code = "authentication_error", status = 401 }
  end
  if resp.status == 426 then
    return "done", { message = "free-tier client upgrade required", code = "authentication_error", status = 426 }
  end
  if resp.status == 429 then
    local lower_body = body_str:lower()
    local quota = lower_body:find("freeusagelimiterror", 1, true) ~= nil
      or lower_body:find("quota", 1, true) ~= nil
      or lower_body:find("free usage", 1, true) ~= nil
    mark_exit_limited(px, quota and "quota" or "rate_limit")
    if quota then
      return "proxy", { message = "free-tier quota exhausted", code = "insufficient_quota", status = 429 }
    end
    return "proxy", { message = "free tier rate limited on current exit", code = "rate_limit", status = 429 }
  end
  if resp.status == 403 then
    -- Blocked egress address: the exit is at fault, not the request. Mark it
    -- for an hour and move to the next one.
    denied = denied + 1
    debug_log("FORBIDDEN", "exit=" .. exit_key(px) .. " body=" .. body_str:sub(1, 200))
    mark_exit_limited(px, "forbidden", FORBIDDEN_TTL)
    return "proxy", { message = "free tier denied the current exit", code = "server_error", status = 403 }
  end
  if resp.status == 400 or resp.status == 404 or resp.status == 422 then
    return "done",
      {
        message = "free tier rejected the request with status " .. tostring(resp.status),
        code = "invalid_request_error",
        status = resp.status,
      }
  end
  return "done",
    { message = "free tier returned status " .. tostring(resp.status), code = "server_error", status = resp.status }
end

-- Dynamic model catalog. Zen /models supplies the membership whitelist
-- (ids only); models.opencode.ai/api.json enriches whitelisted ids with
-- limits, modalities and reasoning flags, and enrichment for ids outside
-- the whitelist is discarded. Rows cache in plugin storage for a day: a
-- miss or an expired entry forces a refresh, and any fetch or parse
-- failure serves an empty list, never stale or hardcoded rows.
local MODEL_CACHE_SCOPE = "model_catalog"
local MODEL_CACHE_TTL = 86400
-- Router-required constants the catalog does not provide, plus the
-- conservative fallback for whitelisted ids the catalog does not enrich.
local MODEL_CACHE_RPM = 60
local MODEL_CACHE_TPM = 100000
local MODEL_CACHE_RPD = 500
local MODEL_DEFAULT_CONTEXT_WINDOW = 200000
local MODEL_DEFAULT_MAX_TOKENS = 32000

-- The catalog fetch is anonymous upstream (UA opencode/latest/<ver>/cli,
-- no auth or session headers), so it uses a bare client, not opencode_headers.
local function string_list(value, fallback)
  if type(value) ~= "table" then return fallback end
  local out = {}
  for i = 1, #value do
    if type(value[i]) == "string" and value[i] ~= "" then table.insert(out, value[i]) end
  end
  if #out == 0 then return fallback end
  return out
end

-- One router-shaped row from a whitelisted id plus its enrichment entry
-- (nil when the catalog lacks the id, which keeps the safe defaults).
local function build_model_row(name, entry)
  local row = { name = name, display_name = name }
  local limit = {}
  if type(entry) == "table" and type(entry.limit) == "table" then limit = entry.limit end
  local context_window = math.floor(tonumber(limit.context) or 0)
  if context_window <= 0 then context_window = MODEL_DEFAULT_CONTEXT_WINDOW end
  row.context_window = context_window
  local max_tokens = math.floor(tonumber(limit.output) or 0)
  if max_tokens <= 0 then max_tokens = MODEL_DEFAULT_MAX_TOKENS end
  row.max_tokens = max_tokens
  row.rpm = MODEL_CACHE_RPM
  row.tpm = MODEL_CACHE_TPM
  row.rpd = MODEL_CACHE_RPD
  row.supported_parameters = { "tools", "tool_choice", "response_format", "temperature", "top_p", "max_tokens" }
  local modalities = {}
  if type(entry) == "table" and type(entry.modalities) == "table" then modalities = entry.modalities end
  row.input_modalities = string_list(modalities.input, { "text" })
  row.output_modalities = string_list(modalities.output, { "text" })
  row.endpoints = { "chat/completions" }
  if type(entry) == "table" and entry.reasoning == true then row.reasoning = { default_enabled = true } end
  return row
end

local function refresh_model_catalog()
  local client = llm_router.http_client({})
  local resp, err = client:request({
    method = "GET",
    url = BASE_URL .. "/models",
    headers = { ["User-Agent"] = catalog_ua(), ["Accept"] = "application/json" },
  })
  if err or resp.status ~= 200 then return nil end
  local ok, parsed = pcall(json.decode, resp.body)
  if not ok or not parsed or type(parsed.data) ~= "table" then return nil end
  local whitelist = {}
  for _, m in ipairs(parsed.data) do
    if type(m) == "table" and type(m.id) == "string" and is_free_model(m.id) then whitelist[m.id] = true end
  end
  local enriched = {}
  local catalog_client = llm_router.http_client({ timeout_ms = 60000 })
  local eres, eerr = catalog_client:request({
    method = "GET",
    url = MODELS_DEV_URL,
    headers = { ["User-Agent"] = catalog_ua(), ["Accept"] = "application/json" },
  })
  if not eerr and eres.status == 200 then
    local ok2, catalog = pcall(json.decode, eres.body)
    if ok2 and type(catalog) == "table" and type(catalog.opencode) == "table" then
      if type(catalog.opencode.models) == "table" then enriched = catalog.opencode.models end
    end
  end
  -- Sorted names keep the served catalog deterministic across refreshes.
  local names = {}
  for name, _ in pairs(whitelist) do
    if type(name) == "string" and name ~= "" then table.insert(names, name) end
  end
  table.sort(names)
  local rows = {}
  for _, name in ipairs(names) do
    table.insert(rows, build_model_row(name, enriched[name]))
  end
  -- An empty refresh stores nothing, so the next call retries instead of
  -- pinning an empty catalog for the TTL. A failed cache write still
  -- serves the fresh rows to this caller. Nil (never an empty table: the
  -- router rejects {} but maps nil to an empty model list) reports failure.
  if #rows == 0 then return nil end
  pcall(llm_router.storage.set, MODEL_CACHE_SCOPE, "infos", rows, { ttl = MODEL_CACHE_TTL })
  return rows
end

-- Responses requires call_id values no longer than 64 characters.
-- Keep native ids when possible; deterministically shorten oversized ids so
-- function_call and function_call_output references remain identical.
local CALL_ID_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
local function normalize_call_id(raw)
  if type(raw) ~= "string" or raw == "" then return "call_unknown" end
  if #raw <= 64 then return raw end
  local ok, hashed = pcall(llm_router.uuid_v5, CALL_ID_NAMESPACE, raw)
  if ok and type(hashed) == "string" and hashed ~= "" then
    local normalized = "call_" .. hashed
    print(
      "[OPENCODE-FREE-DEBUG] normalized oversized call_id length=" .. tostring(#raw) .. " -> " .. tostring(#normalized)
    )
    return normalized
  end
  local normalized = raw:sub(1, 64)
  print("[OPENCODE-FREE-DEBUG] truncated oversized call_id length=" .. tostring(#raw) .. " -> 64")
  return normalized
end

-- Upstream reasoning blobs persist per conversation (10-minute TTL): the
-- router speaks chat protocol, so encrypted_content never arrives in the
-- request. The assembler stores each completed turn's blobs; the builder
-- replays them FIFO against assistant turns carrying reasoning text.
local BLOB_SCOPE = "reasoning_blobs"
local BLOB_TTL = 600

local function blob_store_key(request, model) return conv_key(request, model) end

local function blob_push(request, model, blobs)
  if #blobs == 0 then return end
  pcall(llm_router.storage.set, BLOB_SCOPE, blob_store_key(request, model), blobs, { ttl = BLOB_TTL })
end

local function blob_take_all(request, model)
  local ok, blobs = pcall(llm_router.storage.get, BLOB_SCOPE, blob_store_key(request, model))
  if ok and type(blobs) == "table" then return blobs end
  return {}
end

-- Build the OpenAI Responses input from chat messages. Reasoning items
-- replay stored encrypted blobs FIFO; without a blob the item carries an
-- empty summary, the closest shape the chat protocol allows.
local function build_responses_input(messages, blobs)
  local input = {}
  local blob_idx = 1
  for _, m in ipairs(messages or {}) do
    local role = m.role
    if role == "system" then
      local text = message_text(m)
      if text ~= "" then table.insert(input, { role = "system", content = text }) end
    elseif role == "user" then
      local text = message_text(m)
      if text ~= "" then table.insert(input, { role = "user", content = { { type = "input_text", text = text } } }) end
    elseif role == "assistant" then
      -- Reasoning text round-trips so upstream reuses prior thinking
      -- instead of re-reasoning every turn. Responses carry no encrypted
      -- content through the chat protocol, so only the summary reshapes.
      local rc = m.reasoning_content
      if type(rc) ~= "string" or rc == "" then rc = m.reasoning end
      if type(rc) == "string" and rc ~= "" then
        debug_log("REASONING_PASSTHROUGH", "chars=" .. tostring(#rc))
        local blob = blobs[blob_idx]
        blob_idx = blob_idx + 1
        if type(blob) == "string" and blob ~= "" then
          table.insert(input, { type = "reasoning", encrypted_content = blob, summary = json.decode("[]") })
        else
          table.insert(input, { type = "reasoning", summary = json.decode("[]") })
        end
      end
      local text = message_text(m)
      if text ~= "" then
        table.insert(
          input,
          { role = "assistant", content = { { type = "output_text", text = text } }, phase = "commentary" }
        )
      end
      for _, tc in ipairs(m.tool_calls or {}) do
        local name = tc["function"] and tc["function"].name or ""
        if name ~= "" then
          local args = tc["function"].arguments or "{}"
          if args == "" then args = "{}" end
          table.insert(
            input,
            { type = "function_call", call_id = normalize_call_id(tc.id), name = name, arguments = args }
          )
        end
      end
    elseif role == "tool" then
      table.insert(
        input,
        { type = "function_call_output", call_id = normalize_call_id(m.tool_call_id), output = message_text(m) }
      )
    end
  end
  if #input == 0 then table.insert(input, { role = "user", content = { { type = "input_text", text = "ping" } } }) end
  return input
end

local function build_responses_tools(tools)
  local out = {}
  for _, t in ipairs(normalize_tools(tools)) do
    local name = t.name or ""
    local desc = t.description or ""
    local params = t.parameters
    if t["function"] then
      if t["function"].name and t["function"].name ~= "" then name = t["function"].name end
      if t["function"].description and t["function"].description ~= "" then desc = t["function"].description end
      if t["function"].parameters then params = t["function"].parameters end
    end
    if name ~= "" then
      if params == nil then params = { type = "object", properties = {}, required = json.decode("[]") } end
      table.insert(out, { type = "function", name = name, description = desc, parameters = params })
    end
  end
  return out
end

-- Anonymous chat payload in the genuine client's shape: the fetched agent
-- head leads every turn (no title path), temperature is never sent, tools
-- merge client definitions over the injected built-ins.
local function build_anon_chat_payload(request, model)
  local normalized_tools = normalize_tools(request.tools)
  local head, err = agent_prompt(model)
  if head == nil then return nil, err end
  local messages = { { role = "system", content = head } }
  for _, m in ipairs(request.messages or {}) do
    table.insert(messages, m)
  end
  local payload = {
    model = model,
    messages = messages,
    stream = true,
    stream_options = { include_usage = true },
  }
  payload.max_tokens = output_token_budget(request)
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  local merged, merr = merge_tools(request.tools)
  if merged == nil then return nil, merr end
  if #merged > 0 then
    payload.tools = merged
    if request.tool_choice then
      payload.tool_choice = request.tool_choice
    else
      payload.tool_choice = "auto"
    end
  end
  if type(request.response_format) == "table" then payload.response_format = request.response_format end
  debug_log("CHAT_PAYLOAD", "tools=" .. tostring(#merged) .. " messages=" .. tostring(#messages))
  return payload
end

-- Anonymous Responses payload. reasoning travels only as caller effort;
-- the title-turn default is gone with the title path. prompt_cache_key is
-- the raw session id; include requests encrypted reasoning blobs.
local function build_anon_responses_payload(request, model, ses)
  local normalized_tools = normalize_tools(request.tools)
  local head, err = agent_prompt(model)
  if head == nil then return nil, err end
  local input = { { role = "developer", content = head } }
  for _, item in ipairs(build_responses_input(request.messages, blob_take_all(request, model))) do
    table.insert(input, item)
  end
  local payload = {
    model = model,
    input = input,
    stream = true,
  }
  payload.max_output_tokens = output_token_budget(request)
  payload.store = false
  -- Stable partition across turns: the raw session id, as the genuine client sends it.
  payload.prompt_cache_key = ses
  payload.include = { "reasoning.encrypted_content" }
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
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  local tools_from_builder = build_responses_tools(normalized_tools)
  local merged, merr = merge_tools_resp(tools_from_builder)
  if merged == nil then return nil, merr end
  if #merged > 0 then
    payload.tools = merged
    if request.tool_choice then
      payload.tool_choice = request.tool_choice
    else
      payload.tool_choice = "auto"
    end
  end
  debug_log("RESPONSES_PAYLOAD", "tools=" .. tostring(#(payload.tools or {})) .. " input=" .. tostring(#input))
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
  return payload
end

-- Collect encrypted reasoning blobs from a completed Responses envelope.
local function collect_blobs(raw)
  local blobs = {}
  if type(raw) ~= "table" or type(raw.output) ~= "table" then return blobs end
  for _, item in ipairs(raw.output) do
    if type(item) == "table" and item.type == "reasoning" and type(item.encrypted_content) == "string" then
      table.insert(blobs, item.encrypted_content)
    end
  end
  return blobs
end

-- Assemble one chat.completion from a streamed Responses SSE body.
local function assemble_responses_stream(body, model, request)
  local text_parts = {}
  local reasoning_parts = {}
  local fn_by_index = {}
  local fn_order = {}
  local prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
  local out_model = model
  local terminal_finish = nil
  local blobs = {}
  local function fn_slot(index)
    local acc = fn_by_index[index]
    if not acc then
      acc = { call_id = "", name = "", args = {} }
      fn_by_index[index] = acc
      table.insert(fn_order, index)
    end
    return acc
  end
  for line in (body .. "\n"):gmatch("([^\n]*)\n") do
    if line:sub(1, 6) == "data: " then
      local data = line:sub(7)
      if data ~= "[DONE]" and data ~= "" then
        local ok, ev = pcall(json.decode, data)
        if ok and type(ev) == "table" and type(ev.type) == "string" then
          local et = ev.type
          if et == "response.output_text.delta" and type(ev.delta) == "string" then
            table.insert(text_parts, ev.delta)
          elseif et == "response.reasoning_summary_text.delta" and type(ev.delta) == "string" then
            table.insert(reasoning_parts, ev.delta)
          elseif
            (et == "response.output_item.added" or et == "response.output_item.done")
            and type(ev.item) == "table"
            and ev.item.type == "function_call"
          then
            local acc = fn_slot(ev.output_index or 0)
            if type(ev.item.call_id) == "string" and ev.item.call_id ~= "" then acc.call_id = ev.item.call_id end
            if type(ev.item.name) == "string" and ev.item.name ~= "" then acc.name = ev.item.name end
            if type(ev.item.arguments) == "string" and ev.item.arguments ~= "" then acc.args[1] = ev.item.arguments end
          elseif et == "response.function_call_arguments.delta" and type(ev.delta) == "string" then
            local acc_for_delta = fn_slot(ev.output_index or 0)
            acc_for_delta.args[1] = (acc_for_delta.args[1] or "") .. ev.delta
          elseif et == "response.function_call_arguments.done" then
            -- Arguments complete; assembly reads the accumulated buffer.
          elseif (et == "response.completed" or et == "response.incomplete") and type(ev.response) == "table" then
            local r = ev.response
            if type(r.model) == "string" and r.model ~= "" then out_model = r.model end
            blobs = collect_blobs(r)
            if type(r.usage) == "table" then
              prompt_tokens = r.usage.input_tokens or 0
              completion_tokens = r.usage.output_tokens or 0
              total_tokens = r.usage.total_tokens or (prompt_tokens + completion_tokens)
              local details = r.usage.input_tokens_details
              if type(details) == "table" and tonumber(details.cached_tokens) then
                debug_log(
                  "CACHE",
                  "cached_tokens=" .. tostring(details.cached_tokens) .. " input_tokens=" .. tostring(prompt_tokens)
                )
              end
            end
            if
              et == "response.incomplete"
              and type(r.incomplete_details) == "table"
              and r.incomplete_details.reason == "max_output_tokens"
            then
              terminal_finish = "length"
            end
          end
        end
      end
    end
  end
  blob_push(request, model, blobs)
  table.sort(fn_order)
  local tool_calls = {}
  for _, idx in ipairs(fn_order) do
    local acc = fn_by_index[idx]
    if acc.name ~= "" then
      table.insert(tool_calls, {
        id = acc.call_id,
        type = "function",
        ["function"] = { name = acc.name, arguments = acc.args[1] or "" },
      })
    end
  end
  local finish = terminal_finish or "stop"
  if #tool_calls > 0 then finish = "tool_calls" end
  local tool_calls_final = tool_calls
  if #tool_calls == 0 then tool_calls_final = json.decode("[]") end
  local message = { role = "assistant", content = table.concat(text_parts), tool_calls = tool_calls_final }
  local reasoning = table.concat(reasoning_parts)
  if reasoning ~= "" then message.reasoning_content = reasoning end
  return {
    id = "zen-" .. tostring(os.time()),
    object = "chat.completion",
    created = os.time(),
    model = out_model,
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
                  -- Upstream chat deltas carry the trace as reasoning_content;
                  -- older shapes used reasoning. Accept both.
                  local reasoning = delta.reasoning_content
                  if type(reasoning) ~= "string" or reasoning == "" then reasoning = delta.reasoning end
                  if type(reasoning) == "string" and reasoning ~= "" then table.insert(reasoning_parts, reasoning) end
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
                          if type(fn.arguments) == "string" and fn.arguments ~= "" then acc.args[1] = fn.arguments end
                        end
                      end
                    end
                  end
                end
                if type(ch.finish_reason) == "string" and ch.finish_reason ~= "" then finish = ch.finish_reason end
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
        id = acc.id,
        type = "function",
        ["function"] = { name = acc.name, arguments = acc.args[1] or "" },
      })
    end
  end
  if #tool_calls > 0 and finish == "stop" then finish = "tool_calls" end
  local tool_calls_final = tool_calls
  if #tool_calls == 0 then tool_calls_final = json.decode("[]") end
  local message = { role = "assistant", content = table.concat(text_parts), tool_calls = tool_calls_final }
  local reasoning = table.concat(reasoning_parts)
  if reasoning ~= "" then message.reasoning_content = reasoning end
  return {
    id = "zen-" .. tostring(os.time()),
    object = "chat.completion",
    created = os.time(),
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

-- One upstream fetch: returns (body) or (nil, action, err) where action is
-- "retry" (next exit) or "done" (terminal, return at once).
local function fetch_body(client, url, headers, body, model_name, px)
  local resp, err = client:request({
    method = "POST",
    url = url,
    headers = headers,
    body = body,
    proxy_url = px.url,
  })
  if err then return nil, "retry", err end
  if resp.status == 200 then return resp.body end
  local action, terr = map_upstream(resp, model_name, px)
  return nil, action, terr
end

llm_router.register("opencode-free", {
  icon = "https://opencode.ai/favicon.ico",

  proxy_schema = {},

  get_model_infos = function(ctx)
    local cached = llm_router.storage.get(MODEL_CACHE_SCOPE, "infos")
    if type(cached) == "table" and #cached > 0 then return cached end
    return refresh_model_catalog()
  end,

  complete = function(ctx, request)
    local model = request.model_name
    local endpoint = endpoint_for_model(model)
    local client = llm_router.http_client({})
    local last_err = nil
    -- Run-scoped ids: one session per conversation, one request id per run.
    local ses = stable_session(request, model)
    local msg = mint_msg(mint_ms())

    debug_log(
      "COMPLETE",
      "model="
        .. tostring(model)
        .. " full_model="
        .. tostring(request.model)
        .. " endpoint="
        .. tostring(endpoint)
        .. " request_tools="
        .. table_shape(request.tools)
        .. " tools_count="
        .. tostring(#normalize_tools(request.tools))
        .. " messages="
        .. tostring(type(request.messages) == "table" and #request.messages or -1)
        .. " tool_choice="
        .. debug_json(request.tool_choice)
    )

    local exits, skipped = pick_proxies(ctx, EXIT_ATTEMPTS)
    for _, px in ipairs(exits) do
      if endpoint == "/responses" then
        -- Anonymous Responses path: stream upstream like the genuine
        -- client and assemble the event stream into one completion.
        -- Generations over large contexts run long; the wide timeout only
        -- bounds dead connections, progress streams underneath it.
        local resp_client = llm_router.http_client({ timeout_ms = 300000 })
        local payload, perr = build_anon_responses_payload(request, model, ses)
        if payload == nil then return nil, perr end
        local headers = opencode_headers(endpoint, ses, msg)
        local body = debug_json(payload)
        debug_log("HTTP", "POST " .. BASE_URL .. "/responses body_len=" .. tostring(#body))
        local raw_body, action, err = fetch_body(resp_client, BASE_URL .. "/responses", headers, body, model, px)
        if raw_body then return assemble_responses_stream(raw_body, request.model, request) end
        debug_log("HTTP_ERROR", debug_json(err))
        if action == "done" then return nil, err end
        last_err = err
      else
        -- Anonymous chat path: the free-tier gate only serves stream:true
        -- requests shaped like the genuine client, so always stream upstream
        -- and assemble the SSE body into one completion here.
        local payload, perr = build_anon_chat_payload(request, model)
        if payload == nil then return nil, perr end
        local headers = opencode_headers(endpoint, ses, msg)
        local body = debug_json(payload)
        debug_log("HTTP", "POST " .. BASE_URL .. "/chat/completions body_len=" .. tostring(#body))
        local raw_body, action, err = fetch_body(client, BASE_URL .. "/chat/completions", headers, body, model, px)
        if raw_body then return assemble_chat_response(raw_body, request.model) end
        debug_log("HTTP_ERROR", debug_json(err))
        if action == "done" then return nil, err end
        last_err = err
      end
    end
    return nil, exhausted_error(last_err, skipped)
  end,

  complete_stream = function(ctx, request, emit)
    local model = request.model_name
    local endpoint = endpoint_for_model(model)
    debug_log(
      "STREAM",
      "model="
        .. tostring(model)
        .. " full_model="
        .. tostring(request.model)
        .. " endpoint="
        .. tostring(endpoint)
        .. " request_tools="
        .. table_shape(request.tools)
        .. " tools_count="
        .. tostring(#normalize_tools(request.tools))
        .. " messages="
        .. tostring(type(request.messages) == "table" and #request.messages or -1)
    )
    local client = llm_router.http_client({})
    local full_model = request.model
    local last_err = nil
    local ses = stable_session(request, model)
    local msg = mint_msg(mint_ms())

    local exits, skipped = pick_proxies(ctx, EXIT_ATTEMPTS)
    for _, px in ipairs(exits) do
      local done = false
      local fatal = false

      if endpoint == "/chat/completions" then
        local payload, perr = build_anon_chat_payload(request, model)
        if payload == nil then return nil, perr end
        local headers = opencode_headers(endpoint, ses, msg)
        local body = debug_json(payload)
        debug_log("HTTP_STREAM", "POST " .. BASE_URL .. "/chat/completions body_len=" .. tostring(#body))
        local _, stream_err = client:stream({
          method = "POST",
          url = BASE_URL .. "/chat/completions",
          headers = headers,
          body = body,
          proxy_url = px.url,
          on_response = function(r)
            debug_log("HTTP_STREAM_RESPONSE", "status=" .. tostring(r.status) .. " body=" .. tostring(r.body or ""))
            if r.status ~= 200 then
              local action, terr = map_upstream(r, model, px)
              if action == "done" then fatal = true end
              return terr
            end
          end,
          on_line = function(line)
            debug_log("SSE", "line=" .. tostring(line))
            if line:sub(1, 6) ~= "data: " then return end
            local data = line:sub(7)
            if data == "[DONE]" or data == "" then return end
            local ok, chunk = pcall(json.decode, data)
            if not ok or type(chunk) ~= "table" then return end
            local has_choices = type(chunk.choices) == "table" and #chunk.choices > 0
            if not has_choices and chunk.usage == nil then return end
            chunk.model = full_model
            emit(chunk)
            done = true
          end,
        })
        if stream_err then
          debug_log("HTTP_STREAM_ERROR", debug_json(stream_err))
          last_err = stream_err
          if fatal then return nil, last_err end
          if done then return nil, last_err end
        else
          debug_log("HTTP_STREAM_DONE", "chat stream completed")
          return
        end
      end

      if endpoint == "/responses" then
        -- Anonymous Responses stream: forward upstream SSE as chat chunks
        -- incrementally, like the chat path. Wide timeout: generations over
        -- large contexts run long, progress streams underneath it.
        local stream_client = llm_router.http_client({ timeout_ms = 300000 })
        local payload, perr = build_anon_responses_payload(request, model, ses)
        if payload == nil then return nil, perr end
        local headers = opencode_headers(endpoint, ses, msg)
        local body = debug_json(payload)
        debug_log("HTTP_STREAM", "POST " .. BASE_URL .. "/responses body_len=" .. tostring(#body))
        local fn_state = {}
        local usage = nil
        local finish = "stop"
        local saw_tools = false
        local blobs = {}
        local function fn_acc(index)
          local acc = fn_state[index]
          if not acc then
            acc = { call_id = "", name = "", args = "", emitted = false }
            fn_state[index] = acc
          end
          return acc
        end
        local _, stream_err = stream_client:stream({
          method = "POST",
          url = BASE_URL .. "/responses",
          headers = headers,
          body = body,
          proxy_url = px.url,
          on_response = function(r)
            debug_log("HTTP_STREAM_RESPONSE", "status=" .. tostring(r.status))
            if r.status ~= 200 then
              local action, terr = map_upstream(r, model, px)
              if action == "done" then fatal = true end
              return terr
            end
          end,
          on_line = function(line)
            if line:sub(1, 6) ~= "data: " then return end
            local data = line:sub(7)
            if data == "[DONE]" or data == "" then return end
            local ok, ev = pcall(json.decode, data)
            if not ok or type(ev) ~= "table" or type(ev.type) ~= "string" then return end
            local et = ev.type
            -- Accepted lifecycle events without payload effects.
            if
              et == "response.created"
              or et == "response.in_progress"
              or et == "response.content_part.added"
              or et == "response.content_part.done"
              or et == "response.function_call_arguments.done"
            then
              return
            end
            if et == "response.output_text.delta" and type(ev.delta) == "string" and ev.delta ~= "" then
              emit({
                model = full_model,
                choices = { { index = 0, delta = { role = "assistant", content = ev.delta } } },
              })
              done = true
            elseif et == "response.reasoning_summary_text.delta" and type(ev.delta) == "string" and ev.delta ~= "" then
              emit({
                model = full_model,
                choices = { { index = 0, delta = { role = "assistant", reasoning_content = ev.delta } } },
              })
              done = true
            elseif
              (et == "response.output_item.added" or et == "response.output_item.done")
              and type(ev.item) == "table"
              and ev.item.type == "function_call"
            then
              local acc = fn_acc(ev.output_index or 0)
              if type(ev.item.call_id) == "string" and ev.item.call_id ~= "" then acc.call_id = ev.item.call_id end
              if type(ev.item.name) == "string" and ev.item.name ~= "" then acc.name = ev.item.name end
              if type(ev.item.arguments) == "string" and ev.item.arguments ~= "" then acc.args = ev.item.arguments end
              if et == "response.output_item.done" and acc.name ~= "" and not acc.emitted then
                acc.emitted = true
                saw_tools = true
                emit({
                  model = full_model,
                  choices = {
                    {
                      index = 0,
                      delta = {
                        role = "assistant",
                        tool_calls = {
                          {
                            id = acc.call_id,
                            type = "function",
                            ["function"] = { name = acc.name, arguments = acc.args },
                          },
                        },
                      },
                    },
                  },
                })
                done = true
              end
            elseif et == "response.function_call_arguments.delta" and type(ev.delta) == "string" then
              local acc_for_delta = fn_acc(ev.output_index or 0)
              acc_for_delta.args = acc_for_delta.args .. ev.delta
            elseif (et == "response.completed" or et == "response.incomplete") and type(ev.response) == "table" then
              local r = ev.response
              blobs = collect_blobs(r)
              if type(r.usage) == "table" then
                usage = {
                  prompt_tokens = r.usage.input_tokens or 0,
                  completion_tokens = r.usage.output_tokens or 0,
                  total_tokens = r.usage.total_tokens or 0,
                }
                local details = r.usage.input_tokens_details
                if type(details) == "table" and tonumber(details.cached_tokens) then
                  debug_log(
                    "CACHE",
                    "cached_tokens="
                      .. tostring(details.cached_tokens)
                      .. " input_tokens="
                      .. tostring(usage.prompt_tokens)
                  )
                end
              end
              if et == "response.incomplete" then finish = "length" end
            end
          end,
        })
        blob_push(request, model, blobs)
        if stream_err then
          debug_log("HTTP_STREAM_ERROR", debug_json(stream_err))
          last_err = stream_err
          if fatal then return nil, last_err end
          if done then return nil, last_err end
        else
          -- Terminal chunk: finish reason plus usage. Tool calls emitted
          -- above stay out; anything accumulated but unemitted rides here.
          local order = {}
          for idx, acc in pairs(fn_state) do
            if acc.name ~= "" and not acc.emitted then table.insert(order, idx) end
          end
          table.sort(order)
          local pending_calls = {}
          for _, idx in ipairs(order) do
            local acc = fn_state[idx]
            saw_tools = true
            table.insert(pending_calls, {
              id = acc.call_id,
              type = "function",
              ["function"] = { name = acc.name, arguments = acc.args },
            })
          end
          local terminal = { index = 0, delta = { role = "assistant" }, finish_reason = finish }
          if #pending_calls > 0 then
            terminal.delta.tool_calls = pending_calls
            terminal.finish_reason = "tool_calls"
          elseif saw_tools then
            terminal.finish_reason = "tool_calls"
          end
          local chunk = { model = full_model, choices = { terminal } }
          if usage then chunk.usage = usage end
          emit(chunk)
          debug_log("HTTP_STREAM_DONE", "responses stream completed")
          return
        end
      end
    end
    return nil, exhausted_error(last_err, skipped)
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
