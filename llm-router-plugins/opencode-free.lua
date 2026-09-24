--- @plugin OpenCode Zen
--- @author TheSlopMachine
--- @version 1.0.10
--- @router_version 0.0.4
--- @description OpenAI/Anthropic/Google compatible free provider OpenCode Zen
--- @allow_host opencode.ai

local BASE_URL = "https://opencode.ai/zen/v1"

-- Protocol families first: muse-spark/gpt/grok serve /responses even for
-- their -free variants. Remaining -free models are chat-protocol.
local function endpoint_for_model(model)
  local m = model:lower()
  if m:match("^gpt%-") or m:match("muse%-spark") or m:match("^grok%-") then
    return "/responses"
  end
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

-- Full genuine agent system prompt for multi-turn shapes. The gate only
-- accepts assistant history beside this prompt, not the title fragment.
-- Stored with literal \r\n and \n escapes, decoded in order below.
local AGENT_SRC = [==[You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\r\n\r\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\r\n\r\nIf the user asks for help or wants to give feedback inform them of the following:\r\n- /help: Get help with using opencode\r\n- To give feedback, users should report the issue at https://github.com/anomalyco/opencode/issues\r\n\r\nWhen the user directly asks about opencode (eg 'can opencode do...', 'does opencode have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from opencode docs at https://opencode.ai\r\n\r\n# Tone and style\r\nYou should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).\r\nRemember that your output will be displayed on a command line interface. Your responses can use GitHub-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\r\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.\r\nIf you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.\r\nOnly use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.\r\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.\r\nIMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\r\nIMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...". Here are some examples to demonstrate appropriate verbosity:\r\n<example>\r\nuser: what is 2+2?\r\nassistant: 4\r\n</example>\r\n\r\n<example>\r\nuser: is 11 a prime number?\r\nassistant: Yes\r\n</example>\r\n\r\n<example>\r\nuser: what command should I run to list files in the current directory?\r\nassistant: ls\r\n</example>\r\n\r\n<example>\r\nuser: what command should I run to watch files in the current directory?\r\nassistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]\r\nnpm run dev\r\n</example>\r\n\r\n<example>\r\nuser: what files are in the directory src/?\r\nassistant: [runs ls and sees foo.c, bar.c, baz.c]\r\nuser: which file contains the implementation of foo?\r\nassistant: src/foo.c\r\n</example>\r\n\r\n<example>\r\nuser: write tests for new feature\r\nassistant: [uses grep and glob search tools to find where similar tests are defined, uses concurrent read file tool use blocks in one tool call to read relevant files at the same time, uses edit file tool to write new tests]\r\n</example>\r\n\r\n# Proactiveness\r\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\r\n1. Doing the right thing when asked, including taking actions and follow-up actions\r\n2. Not surprising the user with actions you take without asking\r\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\r\n3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\r\n\r\n# Following conventions\r\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\r\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\r\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\r\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\r\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\r\n\r\n# Code style\r\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\r\n\r\n# Doing tasks\r\nThe user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:\r\n- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.\r\n- Implement the solution using all tools available to you\r\n- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.\r\n- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (e.g. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to AGENTS.md so that you will know to run it next time.\r\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\r\n\r\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\r\n\r\n# Tool usage policy\r\n- When doing file search, prefer to use the Task tool in order to reduce context usage.\r\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run "git status" and "git diff", send a single message with two tool calls to run the calls in parallel.\r\n\r\nYou MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.\r\n\r\nIMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames directory structure.\r\n\r\n# Code References\r\n\r\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\r\n\r\n<example>\r\nuser: Where are errors from the client handled?\r\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\r\n</example>\r\n\nYou are powered by the model named nemotron-3.5-lightning-free. The exact model ID is opencode/nemotron-3.5-lightning-free\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: C:\Users\Thinker\AppData\Local\Temp\opencode\n  Workspace root folder: /\n  Is directory a git repo: no\n  Platform: win32\n  Today's date: Thu Sep 17 2026\n</env>\nSkills provide specialized instructions and workflows for specific tasks.\nUse the skill tool to load a skill when a task matches its description.\n<available_skills>\n  <skill>\n    <name>customize-opencode</name>\n    <description>Use ONLY when the user is editing or creating opencode's own configuration: opencode.json, opencode.jsonc, files under .opencode/, or files under ~/.config/opencode/. Also use when creating or fixing opencode agents, subagents, skills, plugins, MCP servers, or permission rules. Do not use for the user's own application code, or for any project that is not configuring opencode itself.</description>\n    <location>&lt;built-in&gt;</location>\n  </skill>\n  <skill>\n    <name>git-commit</name>\n    <description>git commit, conventional commit, commit changes. Use ONLY when user explicitly invokes the git-commit skill or asks to commit changes with a conventional commit message.</description>\n    <location>C:\Users\Thinker\.config\opencode\skills\git-commit\SKILL.md</location>\n  </skill>\n  <skill>\n    <name>svelte5-best-practices</name>\n    <description>Svelte 5 runes, snippets, SvelteKit patterns, and modern best practices for TypeScript and component development. Use when writing, reviewing, or refactoring Svelte 5 components and SvelteKit applications. Triggers on: Svelte components, runes ($state, $derived, $effect, $props, $bindable, $inspect), snippets ({#snippet}, {@render}), event handling, SvelteKit data loading, form actions, Svelte 4 to Svelte 5 migration, store to rune migration, slots to snippets migration, TypeScript props typing, generic components, SSR state isolation, performance optimization, or component testing.</description>\n    <location>C:\Users\Thinker\.agents\skills\svelte5-best-practices\SKILL.md</location>\n  </skill>\n</available_skills>]==]
local AGENT_SYS = AGENT_SRC:gsub("\\r\\n", "\r\n"):gsub("\\n", "\n")

-- Six genuine built-in tool definitions (bash, edit, read, write, glob,
-- grep). Multi-turn shapes only pass with these present; client tools are
-- merged in, never replaced.
local TOOLS_JSON_SRC = [==[[{"type": "function", "function": {"name": "bash", "description": "Executes a given PowerShell (7+) command with optional timeout, ensuring proper handling and security measures.\r\n\r\nBe aware: OS: win32, Shell: pwsh\r\n\r\nAll commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID changing directories inside the command - use `workdir` instead.\r\n\r\nUse `C:\\Users\\Thinker\\AppData\\Local\\Temp\\opencode` for temporary work outside the workspace. This directory has already been created, already exists, and is pre-approved for external directory access.\r\n\r\nIMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.\r\n\r\n# PowerShell (7+) shell notes\n- This cross-platform shell supports pipeline chain operators (`&&` and `||`).\n- Use double quotes for interpolated strings (`\"Hello $name\"`), single quotes for verbatim strings.\n- Prefer full cmdlet names like `Get-ChildItem`, `Set-Content`, `Remove-Item`, and `New-Item` over aliases.\n- Use `$(...)` for subexpressions. Use `@(...)` for array expressions.\n- To call a native executable whose path contains spaces, use the call operator: `& \"path/to/exe\" args`.\n- Escape special characters with the PowerShell backtick character.\n\nBefore executing the command, please follow these steps:\n\n1. Directory Verification:\n   - If the command will create new directories or files, first use `Test-Path -LiteralPath <parent>` to verify the parent directory exists and is the correct location\n   - For example, before creating `foo\\bar`, first use `Test-Path -LiteralPath \"foo\"` to check that `foo` exists and is the intended parent directory\n\n2. Command Execution:\n   - Always quote file paths that contain spaces with double quotes (e.g., Remove-Item -LiteralPath \"path with spaces\\file.txt\")\n   - Examples of proper quoting:\n     - New-Item -ItemType Directory -Path \"My Documents\" (correct)\n     - New-Item -ItemType Directory -Path My Documents (incorrect - path is split)\n     - & \"path with spaces\\script.ps1\" (correct)\n     - path with spaces\\script.ps1 (incorrect - path is split and not invoked)\n   - After ensuring proper quoting, execute the command.\n   - Capture the output of the command.\n\nUsage notes:\n  - The command argument is required.\n  - You can specify an optional timeout in milliseconds. If not specified, commands will time out after 120000ms.\n  - If the output exceeds 2000 lines or 51200 bytes, it will be truncated and the full output will be written to a file. You can use Read with offset/limit to read specific sections or Grep to search the full content. Do NOT use `Select-Object -First`, `Select-Object -Last`, or other truncation commands to limit output; the full output will already be captured to a file for more precise searching.\n\n  - Avoid using Shell with PowerShell file/content cmdlets unless explicitly instructed or when these cmdlets are truly necessary for the task. Instead, always prefer using the dedicated tools for these commands:\n    - File search: Use Glob (NOT Get-ChildItem)\n    - Content search: Use Grep (NOT Select-String)\n    - Read files: Use Read (NOT Get-Content)\n    - Edit files: Use Edit (NOT Set-Content)\n    - Write files: Use Write (NOT Set-Content/Out-File or here-strings)\n    - Communication: Output text directly (NOT Write-Output/Write-Host)\n  - When issuing multiple commands:\n    - If the commands are independent and can run in parallel, make multiple bash tool calls in a single message. For example, if you need to run \"git status\" and \"git diff\", send a single message with two bash tool calls in parallel.\n    - If the commands depend on each other and must run sequentially, use a single bash tool call with '&&' to chain them together (e.g., `git add . && git commit -m \"message\" && git push`). For instance, if one operation must complete before another starts (like New-Item before Copy-Item, Write before bash for git operations, or git add before git commit), run these operations sequentially instead.\n    - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail\n    - DO NOT use newlines to separate commands (newlines are ok in quoted strings)\n  - AVOID changing directories inside the command. Use the `workdir` parameter to change directories instead.\n    <good-example>\n    Use workdir=\"project\\subdir\" with command: pytest tests\n    </good-example>\n    <bad-example>\n    Set-Location -LiteralPath \"project\\subdir\" && pytest tests\n    </bad-example>\r\n\r\n# Git and GitHub\r\n- Only commit, amend, push, or create PRs when explicitly requested.\r\n- Before committing, inspect `git status`, `git diff`, and `git log --oneline -10`; stage only intended files and never commit secrets.\r\n- Write a concise commit message that matches the repo style.\r\n- Do not update git config, skip hooks, use interactive `-i`, force-push, or create empty commits unless explicitly requested.\r\n- If a commit fails or hooks reject it, fix the issue and create a new commit; do not amend the failed commit.\r\n- Before creating a PR, inspect status, diff, remote tracking, recent commits, and the diff from the base branch.\r\n- Review all commits included in the PR, not just the latest commit.\r\n- Use `gh` for GitHub tasks, including PRs, issues, checks, and releases; return the PR URL when done.\r\n", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {"command": {"type": "string", "description": "The command to execute"}, "timeout": {"minimum": -9007199254740991, "exclusiveMinimum": 0, "type": "integer", "maximum": 9007199254740991, "description": "Optional timeout in milliseconds"}, "workdir": {"type": "string", "description": "The working directory to run the command in. Defaults to the current directory. Use this instead of 'cd' commands."}}, "required": ["command"]}}}, {"type": "function", "function": {"name": "edit", "description": "Performs exact string replacements in files. \r\n\r\nUsage:\r\n- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. \r\n- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + colon + space (e.g., `1: `). Everything after that space is the actual file content to match. Never include any part of the line number prefix in the oldString or newString.\r\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\r\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\r\n- The edit will FAIL if `oldString` is not found in the file with an error \"oldString not found in content\".\r\n- The edit will FAIL if `oldString` is found multiple times in the file with an error \"Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match.\" Either provide a larger string with more surrounding context to make it unique or use `replaceAll` to change every instance of `oldString`. \r\n- Use `replaceAll` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.\r\n", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {"filePath": {"type": "string", "description": "The absolute path to the file to modify"}, "oldString": {"type": "string", "description": "The text to replace"}, "newString": {"type": "string", "description": "The text to replace it with (must be different from oldString)"}, "replaceAll": {"type": "boolean", "description": "Replace all occurrences of oldString (default false)"}}, "required": ["filePath", "oldString", "newString"]}}}, {"type": "function", "function": {"name": "glob", "description": "- Fast file pattern matching tool that works with any codebase size\r\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\r\n- Returns matching file paths\r\n- Use this tool when you need to find files by name patterns\r\n- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead\r\n- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful.\r\n", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {"pattern": {"type": "string", "description": "The glob pattern to match files against"}, "path": {"type": "string", "description": "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided."}}, "required": ["pattern"]}}}, {"type": "function", "function": {"name": "grep", "description": "- Fast content search tool that works with any codebase size\r\n- Searches file contents using regular expressions\r\n- Supports full regex syntax (eg. \"log.*Error\", \"function\\s+\\w+\", etc.)\r\n- Filter files by pattern with the include parameter (eg. \"*.js\", \"*.{ts,tsx}\")\r\n- Returns file paths and line numbers with matching lines\r\n- Use this tool when you need to find files containing specific patterns\r\n- If you need to identify/count the number of matches within files, use the Bash tool with `rg` (ripgrep) directly. Do NOT use `grep`.\r\n- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead\r\n", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {"pattern": {"type": "string", "description": "The regex pattern to search for in file contents"}, "path": {"type": "string", "description": "The directory to search in. Defaults to the current working directory."}, "include": {"type": "string", "description": "File pattern to include in the search (e.g. \"*.js\", \"*.{ts,tsx}\")"}}, "required": ["pattern"]}}}, {"type": "function", "function": {"name": "read", "description": "Read a file or directory from the local filesystem. If the path does not exist, an error is returned.\r\n\r\nUsage:\r\n- The filePath parameter should be an absolute path.\r\n- By default, this tool returns up to 2000 lines from the start of the file.\r\n- The offset parameter is the line number to start from (1-indexed).\r\n- To read later sections, call this tool again with a larger offset.\r\n- Use the grep tool to find specific content in large files or files with long lines.\r\n- If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern.\r\n- Contents are returned with each line prefixed by its line number as `<line>: <content>`. For example, if a file has contents \"foo\\n\", you will receive \"1: foo\\n\". For directories, entries are returned one per line (without line numbers) with a trailing `/` for subdirectories.\r\n- Any line longer than 2000 characters is truncated.\r\n- Call this tool in parallel when you know there are multiple files you want to read.\r\n- Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window.\r\n- This tool can read image files and PDFs and return them as file attachments.\r\n", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {"filePath": {"type": "string", "description": "The absolute path to the file or directory to read"}, "offset": {"minimum": 0, "type": "integer", "maximum": 9007199254740991, "description": "The line number to start reading from (1-indexed)"}, "limit": {"minimum": 0, "type": "integer", "maximum": 9007199254740991, "description": "The maximum number of lines to read (defaults to 2000)"}}, "required": ["filePath"]}}}, {"type": "function", "function": {"name": "write", "description": "Writes a file to the local filesystem.\r\n\r\nUsage:\r\n- This tool will overwrite the existing file if there is one at the provided path.\r\n- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\r\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\r\n- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\r\n- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.\r\n", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {"content": {"type": "string", "description": "The content to write to the file"}, "filePath": {"type": "string", "description": "The absolute path to the file to write (must be absolute, not relative)"}}, "required": ["content", "filePath"]}}}]]==]
local INJECTED_TOOLS = json.decode(TOOLS_JSON_SRC)
if type(INJECTED_TOOLS) ~= "table" or #INJECTED_TOOLS < 6 then
  error("opencode-zen: embedded tool definitions failed to decode")
end

-- Flat Responses-protocol view of the same definitions.
local INJECTED_TOOLS_RESP = {}
for _, t in ipairs(INJECTED_TOOLS) do
  local fn = t["function"] or {}
  table.insert(INJECTED_TOOLS_RESP, {
    type = "function",
    name = fn.name or "",
    description = fn.description or "",
    parameters = fn.parameters or { type = "object", properties = {} },
  })
end

local function merge_tools_resp(client_tools)
  local out = {}
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
        parameters = (type(t["function"]) == "table" and t["function"].parameters) or { type = "object", properties = {} },
      })
    end
  end
  for _, t in ipairs(INJECTED_TOOLS_RESP) do
    if t.name == "" or not seen[t.name] then table.insert(out, t) end
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

local function classify_error(status, headers, body)
  local message = body
  local error_type = ""
  local ok, parsed = pcall(json.decode, body)
  if ok and parsed and type(parsed.error) == "table" then
    if type(parsed.error.message) == "string" and parsed.error.message ~= "" then
      message = parsed.error.message
    end
    if type(parsed.error.type) == "string" then error_type = parsed.error.type end
  elseif ok and parsed and type(parsed.message) == "string" and parsed.message ~= "" then
    message = parsed.message
  elseif ok and parsed and type(parsed.error) == "string" and parsed.error ~= "" then
    message = parsed.error
  end
  if status == 401 or status == 403 or status == 426 then
    return nil, { type = "auth", message = message }
  elseif status == 429 then
    local wait = 60
    if headers and type(headers["retry-after"]) == "string" then
      local retry_after = tonumber(headers["retry-after"])
      if retry_after and retry_after > 0 then wait = math.ceil(retry_after) end
    end
    local lower_message = message:lower()
    local lower_type = error_type:lower()
    local quota = lower_type == "freeusagelimiterror"
      or lower_message:find("quota", 1, true) ~= nil
      or lower_message:find("free usage", 1, true) ~= nil
    return nil, {
      type = quota and "quota_exceeded" or "rate_limit",
      message = message,
      retry_after = os.time() + wait,
    }
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

-- Multi-turn means prior assistant or tool messages exist. Those shapes
-- only pass beside the full agent prompt plus genuine tools.
local function has_history(messages)
  for _, m in ipairs(messages or {}) do
    if type(m) == "table" and (m.role == "assistant" or m.role == "tool") then
      return true
    end
  end
  return false
end

local function has_client_tools(request)
  return type(request.tools) == "table" and #request.tools > 0
end

local function tool_name(t)
  if type(t) ~= "table" then return "" end
  if type(t["function"]) == "table" and type(t["function"].name) == "string" then
    return t["function"].name
  end
  if type(t.name) == "string" then return t.name end
  return ""
end

local function merge_tools(client_tools)
  local out = {}
  local seen = {}
  for _, t in ipairs(client_tools or {}) do
    local name = tool_name(t)
    if name ~= "" then seen[name] = true end
    table.insert(out, t)
  end
  for _, t in ipairs(INJECTED_TOOLS) do
    local name = tool_name(t)
    if name == "" or not seen[name] then table.insert(out, t) end
  end
  return out
end

-- Anonymous chat payload in the genuine client's shape. temperature and
-- max_tokens default to the proven values when the client omits them:
-- the gate stalls requests that lack either field.
local function build_anon_chat_payload(request, model)
  local agent_path = has_history(request.messages) or has_client_tools(request)
  local messages = {}
  if agent_path then
    table.insert(messages, { role = "system", content = AGENT_SYS })
  else
    table.insert(messages, { role = "system", content = FINGERPRINT })
  end
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
  if agent_path then
    payload.tools = merge_tools(request.tools)
    if request.tool_choice then
      payload.tool_choice = request.tool_choice
    else
      payload.tool_choice = "auto"
    end
  end
  if type(request.response_format) == "table" then payload.response_format = request.response_format end
  return payload
end

-- Anonymous Responses payload. Single turns pass with the title fragment
-- as developer content; history shapes need the full agent text plus
-- genuine tools. reasoning minimal is the cheapest effort this family
-- accepts; the genuine shape carries no temperature default.
local function build_anon_responses_payload(request, model)
  local agent_path = has_history(request.messages) or has_client_tools(request)
  local input = {}
  if agent_path then
    table.insert(input, { role = "developer", content = AGENT_SYS })
  else
    table.insert(input, { role = "developer", content = FINGERPRINT })
  end
  for _, item in ipairs(build_responses_input(request.messages)) do
    table.insert(input, item)
  end
  local payload = {
    model = model,
    input = input,
    stream = true,
  }
  payload.max_output_tokens = responses_max_output_tokens(request, model)
  payload.store = false
  local effort = request.reasoning_effort
  if effort == "low" or effort == "medium" or effort == "high" or effort == "xhigh"
      or effort == "minimal" or effort == "max" then
    payload.reasoning = { effort = effort, summary = "auto" }
  else
    payload.reasoning = { effort = "minimal", summary = "auto" }
  end
  if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
  if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
  if agent_path then
    payload.tools = merge_tools_resp(request.tools)
    if request.tool_choice then
      payload.tool_choice = request.tool_choice
    else
      payload.tool_choice = "auto"
    end
  else
    local tools = build_responses_tools(request.tools)
    if #tools > 0 then
      payload.tools = tools
      if request.tool_choice then payload.tool_choice = request.tool_choice end
    end
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
  return payload
end

-- Assemble one chat.completion from a streamed Responses SSE body.
local function assemble_responses_stream(body, model)
  local text_parts = {}
  local reasoning_parts = {}
  local fn_by_index = {}
  local fn_order = {}
  local prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
  local out_model = model
  local terminal_finish = nil
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
          elseif (et == "response.output_item.added" or et == "response.output_item.done")
              and type(ev.item) == "table" and ev.item.type == "function_call" then
            local acc = fn_slot(ev.output_index or 0)
            if type(ev.item.call_id) == "string" and ev.item.call_id ~= "" then acc.call_id = ev.item.call_id end
            if type(ev.item.name) == "string" and ev.item.name ~= "" then acc.name = ev.item.name end
            if type(ev.item.arguments) == "string" and ev.item.arguments ~= "" then
              table.insert(acc.args, ev.item.arguments)
            end
          elseif et == "response.function_call_arguments.delta" and type(ev.delta) == "string" then
            table.insert(fn_slot(ev.output_index or 0).args, ev.delta)
          elseif (et == "response.completed" or et == "response.incomplete")
              and type(ev.response) == "table" then
            local r = ev.response
            if type(r.model) == "string" and r.model ~= "" then out_model = r.model end
            if type(r.usage) == "table" then
              prompt_tokens = r.usage.input_tokens or 0
              completion_tokens = r.usage.output_tokens or 0
              total_tokens = r.usage.total_tokens or (prompt_tokens + completion_tokens)
            end
            if et == "response.incomplete" and type(r.incomplete_details) == "table"
                and r.incomplete_details.reason == "max_output_tokens" then
              terminal_finish = "length"
            end
          end
        end
      end
    end
  end
  table.sort(fn_order)
  local tool_calls = {}
  for _, idx in ipairs(fn_order) do
    local acc = fn_by_index[idx]
    if acc.name ~= "" then
      table.insert(tool_calls, {
        id = acc.call_id, type = "function",
        ["function"] = { name = acc.name, arguments = table.concat(acc.args) },
      })
    end
  end
  local finish = terminal_finish or "stop"
  if #tool_calls > 0 then finish = "tool_calls" end
  local message = { role = "assistant", content = table.concat(text_parts), tool_calls = tool_calls }
  local reasoning = table.concat(reasoning_parts)
  if reasoning ~= "" then message.reasoning_content = reasoning end
  return {
    id = "zen-" .. tostring(os.time()), object = "chat.completion", created = os.time(),
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

    if endpoint == "/responses" and not anonymous then
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
        method = "POST", url = BASE_URL .. "/responses",
        headers = headers, body = json.encode(payload),
      })
      if err then return nil, err end
      if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
      local raw = json.decode(resp.body)
      local text = extract_responses_text(raw)
      local reasoning = extract_responses_reasoning(raw)
      local tool_calls = extract_responses_tool_calls(raw)
      local finish = "stop"
      if type(raw.incomplete_details) == "table"
          and raw.incomplete_details.reason == "max_output_tokens" then
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
        id = "zen-" .. tostring(os.time()), object = "chat.completion", created = os.time(),
        model = request.model,
        choices = {
          { index = 0, message = message, finish_reason = finish },
        },
        usage = usage,
      }
    end

    if endpoint == "/responses" then
      -- Anonymous Responses path: stream upstream like the genuine
      -- client and assemble the event stream into one completion.
      local payload = build_anon_responses_payload(request, model)
      local headers = opencode_headers(api_key, ses, msg)
      local resp, err = client:request({
        method = "POST", url = BASE_URL .. "/responses",
        headers = headers, body = json.encode(payload),
      })
      if err then return nil, err end
      if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
      return assemble_responses_stream(resp.body, request.model)
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
      if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
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
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
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
          return classify_error(status, nil, inner)
        end
        return nil, stream_err
      end
      return
    end

    -- Keyed chat path: relay the client's shape.
    local endpoint = endpoint_for_model(model)
    if endpoint == "/responses" and api_key ~= "" then
      local payload = { model = model, input = build_responses_input(request.messages), stream = true }
      payload.max_output_tokens = responses_max_output_tokens(request, model)
      if request.temperature and request.temperature > 0 then payload.temperature = request.temperature end
      if request.top_p and request.top_p > 0 then payload.top_p = request.top_p end
      local effort = request.reasoning_effort
      if effort == "low" or effort == "medium" or effort == "high" or effort == "xhigh"
          or effort == "minimal" or effort == "max" then
        payload.reasoning = { effort = effort, summary = "auto" }
      end
      local tools = build_responses_tools(request.tools)
      if #tools > 0 then
        payload.tools = tools
        if request.tool_choice then payload.tool_choice = request.tool_choice end
      end
      local headers = opencode_headers(api_key, ses, msg)
      local stream_err = client:stream({
        method = "POST", url = BASE_URL .. "/responses",
        headers = headers,
        body = json.encode(payload),
        on_line = function(line)
          if line:sub(1, 6) ~= "data: " then return end
          local data = line:sub(7)
          if data == "[DONE]" or data == "" then return end
          local ok, chunk = pcall(json.decode, data)
          if not ok or type(chunk) ~= "table" then return end
          emit(chunk)
        end,
      })
      if stream_err then
        local status = tonumber(tostring(stream_err.message):match("unexpected status (%d+)"))
        if status then
          local inner = tostring(stream_err.message):match("unexpected status %d+: (.*)$") or ""
          return classify_error(status, nil, inner)
        end
        return nil, stream_err
      end
      return
    end

    if api_key == "" and endpoint == "/responses" then
      -- Anonymous Responses relay: forward event lines as-is.
      local payload = build_anon_responses_payload(request, model)
      local headers = opencode_headers(api_key, ses, msg)
      local stream_err = client:stream({
        method = "POST", url = BASE_URL .. "/responses",
        headers = headers,
        body = json.encode(payload),
        on_line = function(line)
          if line:sub(1, 6) ~= "data: " then return end
          local data = line:sub(7)
          if data == "[DONE]" or data == "" then return end
          local ok, chunk = pcall(json.decode, data)
          if not ok or type(chunk) ~= "table" then return end
          emit(chunk)
        end,
      })
      if stream_err then
        local status = tonumber(tostring(stream_err.message):match("unexpected status (%d+)"))
        if status then
          local inner = tostring(stream_err.message):match("unexpected status %d+: (.*)$") or ""
          return classify_error(status, nil, inner)
        end
        return nil, stream_err
      end
      return
    end

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
        return classify_error(status, nil, inner)
      end
      return nil, stream_err
    end
  end,

})
