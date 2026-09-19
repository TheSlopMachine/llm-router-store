--- @plugin Google AI Studio
--- @author TheSlopMachine
--- @version 1.6.9
--- @router_version 0.0.6
--- @description Google Gemini models via AI Studio API
--- @allow_host generativelanguage.googleapis.com
--- @proxy_location US
--- @proxy_force_on_mismatch true

local BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

-- Per-model quota memory: an upstream 429 with quota wording is remembered
-- per credential+model in plugin storage. Blocked pairs fail fast without
-- another upstream hit; the router still rotates to the next candidate.
local QUOTA_SCOPE = "model_quota"

-- Cooldown for transient upstream/timeout failures without a retry hint.
local UPSTREAM_COOLDOWN = 120

-- "[model key abc123] " prefix so router logs show provider, model and key.
local function err_prefix(credential, model)
  if model == nil then return "" end
  local cid = (credential and credential.id) or ""
  if #cid > 8 then cid = cid:sub(1, 8) end
  if cid == "" then cid = "?" end
  return "[" .. tostring(model) .. " key " .. cid .. "] "
end

local function quota_storage()
  if llm_router == nil or llm_router.storage == nil then return nil end
  return llm_router.storage
end

local function quota_key(credential, model)
  local cid = (credential and credential.id) or ""
  if cid == "" or model == nil then return nil end
  return cid .. "|" .. tostring(model)
end

-- Remaining block seconds when the pair is still exhausted, else nil.
local function quota_blocked(credential, model)
  local st = quota_storage()
  local key = quota_key(credential, model)
  if st == nil or key == nil then return nil end
  local ok, reset_at = pcall(st.get, QUOTA_SCOPE, key)
  if not ok or reset_at == nil then return nil end
  local reset_n = tonumber(reset_at)
  if reset_n == nil then
    pcall(st.delete, QUOTA_SCOPE, key)
    return nil
  end
  local now = os.time()
  if reset_n > now then return reset_n - now end
  pcall(st.delete, QUOTA_SCOPE, key)
  return nil
end

local function quota_remember(credential, model, wait)
  local st = quota_storage()
  local key = quota_key(credential, model)
  if st == nil or key == nil then return end
  pcall(st.set, QUOTA_SCOPE, key, os.time() + (wait or 60))
end

-- Model-wide cooldown: overload is a property of the upstream model, not the
-- key, so one hot model blocks all keys at once.
local MODEL_SCOPE = "model_cooldown"

local function model_remember(model, wait)
  local st = quota_storage()
  if st == nil or model == nil then return end
  pcall(st.set, MODEL_SCOPE, tostring(model), os.time() + (wait or UPSTREAM_COOLDOWN))
end

-- Remaining model-wide block seconds, else nil.
local function model_blocked(model)
  local st = quota_storage()
  if st == nil or model == nil then return nil end
  local ok, reset_at = pcall(st.get, MODEL_SCOPE, tostring(model))
  if not ok or reset_at == nil then return nil end
  local reset_n = tonumber(reset_at)
  if reset_n == nil then
    pcall(st.delete, MODEL_SCOPE, tostring(model))
    return nil
  end
  local now = os.time()
  if reset_n > now then return reset_n - now end
  pcall(st.delete, MODEL_SCOPE, tostring(model))
  return nil
end

-- Fail-fast error when the model or the pair is still blocked, else nil.
-- Model-wide cooldown wins: it covers every key at once.
local function quota_fail_fast(credential, model)
  local blocked = model_blocked(model)
  if blocked == nil then blocked = quota_blocked(credential, model) end
  if blocked == nil then return nil end
  return { type = "quota_exceeded",
    message = err_prefix(credential, model) .. "cooling down, retry in " .. blocked .. "s",
    retry_after = os.time() + blocked }
end

local function classify_error(status, headers, body, credential, model)
  local message = body
  local ok, parsed = pcall(json.decode, body)
  if ok and parsed and parsed.error and type(parsed.error.message) == "string" and parsed.error.message ~= "" then
    message = parsed.error.message
  end
  message = err_prefix(credential, model) .. message
  if status == 401 or status == 403 then
    return nil, { type = "auth", message = message }
  elseif status == 400 then
    local lower = message:lower()
    if lower:find("location is not supported", 1, true) then
      -- Geo-blocked: the proxy exit is at fault, not the request.
      return nil, { type = "geo", message = message }
    elseif lower:find("api key not valid", 1, true)
        or lower:find("api_key_invalid", 1, true)
        or lower:find("invalid api key", 1, true)
        or lower:find("unauthenticated", 1, true) then
      return nil, { type = "auth", message = message }
    end
  elseif status == 429 then
    local wait = 60
    if headers and type(headers["retry-after"]) == "string" then
      local n = tonumber(headers["retry-after"])
      if n and n > 0 then wait = math.ceil(n) end
    else
      -- Google embeds the delay in the body ("Please retry in 54.74s").
      local lower_msg = message:lower()
      local secs = lower_msg:match("retry in ([%d%.]+)%s*s")
        or lower_msg:match("retry after ([%d%.]+)%s*s")
      local n = secs and tonumber(secs)
      if n and n > 0 then wait = math.ceil(n) end
    end
    -- Any quota wording deprioritizes the credential; pure rate limits just back off.
    local lower = message:lower()
    if lower:find("per day", 1, true) or lower:find("perday", 1, true)
        or lower:find("daily", 1, true) or lower:find("quota", 1, true)
        or lower:find("free_tier", 1, true) or lower:find("free tier", 1, true)
        or lower:find("billing", 1, true) then
      quota_remember(credential, model, wait)
      return nil, { type = "quota_exceeded", message = message, retry_after = os.time() + wait }
    end
    quota_remember(credential, model, wait)
    return nil, { type = "rate_limit", message = message, retry_after = os.time() + wait }
  elseif status == 408 or status == 504 then
    model_remember(model, UPSTREAM_COOLDOWN)
    return nil, { type = "timeout", message = message }
  elseif status >= 500 then
    model_remember(model, UPSTREAM_COOLDOWN)
    return nil, { type = "upstream", message = message }
  end
  return nil, { type = "invalid_request", message = message }
end

-- Re-classify transport-wrapped stream errors ("unexpected status N: body").
local function classify_stream_error(stream_err, credential, model)
  local msg = tostring((type(stream_err) == "table" and stream_err.message) or stream_err)
  local status = tonumber(msg:match("unexpected status (%d+)"))
  if status then
    local body = msg:match("unexpected status %d+: (.*)$") or ""
    return classify_error(status, nil, body, credential, model)
  end
  return nil, stream_err
end

local function api_key_of(credential)
  if credential == nil or credential.data == nil then return "" end
  return credential.data.api_key or ""
end

local function invalid_request(message)
  return { type = "invalid_request", message = message }
end

local function map_finish_reason(r)
  if r == nil or r == "" then return "stop" end
  if r == "STOP" then return "stop"
  elseif r == "MAX_TOKENS" then return "length" end
  -- Every other documented FinishReason is a policy/content block of some
  -- kind; OpenAI has a single content_filter bucket for all of them.
  return "content_filter"
end

-- Plain-text view of message content. Content arrives either as a string or
-- as an array of content parts. Only text is extracted here; binary parts
-- (image_url, input_audio) are handled structurally by user_content.
local function text_of_content(content)
  if type(content) == "string" then return content end
  if type(content) == "number" then return tostring(content) end
  if type(content) ~= "table" then return "" end
  -- Single part object (not an array).
  if type(content.text) == "string" and content.text ~= "" then return content.text end
  if type(content.content) == "string" and content.content ~= "" then return content.content end
  local parts = {}
  for _, p in ipairs(content) do
    if type(p) == "string" then
      if p ~= "" then table.insert(parts, p) end
    elseif type(p) == "table" then
      if type(p.text) == "string" and p.text ~= "" then
        table.insert(parts, p.text)
      elseif type(p.content) == "string" and p.content ~= "" then
        table.insert(parts, p.content)
      elseif type(p.content) == "table" then
        local inner = text_of_content(p.content)
        if inner ~= "" then table.insert(parts, inner) end
      end
    end
  end
  return table.concat(parts, "\n")
end

-- A message whose entire trimmed text is a data: URL carries binary inline.
-- (Fallback path for clients that cannot send image_url parts.)
local function parse_data_url(s)
  if type(s) ~= "string" then return nil end
  local trimmed = s:match("^%s*(.-)%s*$")
  local mime, b64 = trimmed:match("^data:([^;,]+);base64,(.+)$")
  if not mime or not b64 or b64 == "" then return nil end
  local family = mime:match("^([^/]+)/")
  if family ~= "image" and family ~= "audio" and family ~= "video"
      and mime ~= "application/pdf" then
    return nil
  end
  return mime, b64
end

-- Split off system/developer messages; map the rest to Google contents.
-- Returns contents, system_parts, err. Tool results are resolved against
-- assistant tool_calls seen earlier in the same history (two-pass build).
local function build_contents(messages)
  local call_names = {}
  for _, m in ipairs(messages or {}) do
    if m.role == "assistant" and type(m.tool_calls) == "table" then
      for _, tc in ipairs(m.tool_calls) do
        local fn = tc and tc["function"]
        if tc and tc.id and tc.id ~= "" and fn and fn.name and fn.name ~= "" then
          call_names[tc.id] = fn.name
        end
      end
    end
  end

  local system_parts = {}
  local contents = {}
  local current = nil
  local current_merge = false
  local function flush()
    if current ~= nil then
      if #current.parts > 0 then table.insert(contents, current) end
      current = nil
      current_merge = false
    end
  end
  -- Mergeable plain-text block of one role.
  local function text_part(role, text)
    if text == "" then return end
    if current and current.role == role and current_merge then
      table.insert(current.parts, { text = text })
    else
      flush()
      current = { role = role, parts = { { text = text } } }
      current_merge = true
    end
  end
  -- Raw part into a model block (thoughts, text, function calls).
  local function model_part(part)
    if not (current and current.role == "model") then
      flush()
      current = { role = "model", parts = {} }
    end
    current_merge = true
    table.insert(current.parts, part)
  end
  -- A function result always lives in its own user block; consecutive tool
  -- messages share one block.
  local function fnresp_part(fr)
    if current and current.role == "user" and not current_merge then
      table.insert(current.parts, { functionResponse = fr })
    else
      flush()
      current = { role = "user", parts = { { functionResponse = fr } } }
      current_merge = false
    end
  end
  -- Structured walk over user content: text accumulates for merging,
  -- binary parts (image_url / input_audio / bare data: URLs) become raw
  -- inlineData parts in place. Returns err on unsupported remote content.
  local function user_content(role, content, name)
    local texts = {}
    local named = false
    local function flush_text()
      if #texts == 0 then return nil end
      local text = table.concat(texts, "\n")
      texts = {}
      if not named and name and name ~= "" then
        text = "[" .. name .. "] " .. text
        named = true
      end
      text_part(role, text)
      return nil
    end
    local function push_binary(part)
      flush_text()
      if not (current and current.role == role and current_merge) then
        flush()
        current = { role = role, parts = {} }
      end
      current_merge = true
      table.insert(current.parts, part)
      return nil
    end
    local function add_segment(seg)
      if type(seg) == "string" then
        if seg == "" then return nil end
        local mime, b64 = parse_data_url(seg)
        if mime then return push_binary({ inlineData = { mimeType = mime, data = b64 } }) end
        table.insert(texts, seg)
        return nil
      end
      if type(seg) ~= "table" then return nil end
      if type(seg.image_url) == "table" then
        local url = seg.image_url.url or ""
        local mime, b64 = parse_data_url(url)
        if mime then
          -- NOTE: OpenAI detail (low/high/auto) is intentionally not mapped
          -- to Part.mediaResolution: the API rejects the documented level
          -- strings on this endpoint, and resolution is advisory only.
          return push_binary({ inlineData = { mimeType = mime, data = b64 } })
        end
        if url ~= "" then
          return "image URL is not a data: URL; the plugin cannot download remote images"
        end
        return nil
      end
      if type(seg.input_audio) == "table" then
        local data = seg.input_audio.data or ""
        local format = (seg.input_audio.format or ""):lower()
        local mime = nil
        if format == "wav" then mime = "audio/wav"
        elseif format == "mp3" then mime = "audio/mpeg" end
        if data == "" or not mime then
          return "input_audio needs base64 data and format wav or mp3"
        end
        return push_binary({ inlineData = { mimeType = mime, data = data } })
      end
      local text = text_of_content(seg)
      if text ~= "" then table.insert(texts, text) end
      return nil
    end
    if type(content) == "string" or type(content) == "number" then
      local err = add_segment(tostring(content))
      if err then return err end
    elseif type(content) == "table" then
      for _, seg in ipairs(content) do
        local err = add_segment(seg)
        if err then return err end
      end
    end
    flush_text()
    return nil
  end

  for _, m in ipairs(messages or {}) do
    local role = m.role
    if role == "system" or role == "developer" then
      local text = text_of_content(m.content)
      if m.name and m.name ~= "" and text ~= "" then text = "[" .. m.name .. "] " .. text end
      if text ~= "" then table.insert(system_parts, { text = text }) end
    elseif role == "assistant" then
      local text = text_of_content(m.content)
      if (m.refusal == nil or m.refusal == "") and text == ""
          and (not m.tool_calls or #m.tool_calls == 0)
          and (not m.reasoning_content or m.reasoning_content == "") then
        -- skip empty assistant turns (no text, calls, reasoning or refusal)
      else
        if m.reasoning_content and m.reasoning_content ~= "" then
          model_part({ text = m.reasoning_content, thought = true })
        end
        if text ~= "" then model_part({ text = text }) end
        if m.refusal and m.refusal ~= "" and text == ""
            and (not m.tool_calls or #m.tool_calls == 0) then
          model_part({ text = m.refusal })
        end
        for _, tc in ipairs(type(m.tool_calls) == "table" and m.tool_calls or {}) do
          if type(tc) ~= "table" then
            return nil, nil, invalid_request("assistant tool call must be an object")
          end
          local fn = tc["function"] or {}
          if type(fn.name) ~= "string" or fn.name == "" then
            return nil, nil, invalid_request("assistant tool call is missing a function name")
          end
          local args = {}
          if type(fn.arguments) == "string" and fn.arguments ~= "" then
            local ok, parsed = pcall(json.decode, fn.arguments)
            if not ok or type(parsed) ~= "table" then
              return nil, nil, invalid_request("tool call arguments are not valid JSON for function " .. fn.name)
            end
            args = parsed
          elseif type(fn.arguments) == "table" then
            args = fn.arguments
          end
          local fc = { name = fn.name, args = args }
          if tc.id and tc.id ~= "" then fc.id = tc.id end
          local part = { functionCall = fc }
          -- Restore thought_signature from id if present (workaround for Crush not preserving extra_content)
          if tc.id and tc.id ~= "" then
            local pipe = tc.id:find("|")
            if pipe then
              local original_id = tc.id:sub(1, pipe - 1)
              local signature = tc.id:sub(pipe + 1)
              fc.id = original_id
              part.thoughtSignature = signature
            end
          end
          model_part(part)
        end
      end
    elseif role == "tool" then
      local call_id = m.tool_call_id or ""
      local name = call_names[call_id]
      if not name then
        return nil, nil, invalid_request("tool result references unknown tool call id " .. tostring(call_id))
      end
      local raw = text_of_content(m.content)
      local response
      if raw == "" then
        response = {}
      else
        local ok, parsed = pcall(json.decode, raw)
        if ok and type(parsed) == "table" and parsed[1] == nil then
          response = parsed
        else
          response = { output = raw }
        end
      end
      local fr = { name = name, response = response }
      if call_id ~= "" then fr.id = call_id end
      fnresp_part(fr)
    else
      local uerr = user_content("user", m.content, m.name)
      if uerr then return nil, nil, invalid_request(uerr) end
    end
  end
  flush()
  if #contents == 0 then
    return nil, nil, invalid_request("messages produced no content")
  end
  return contents, system_parts, nil
end

-- OpenAI function tools -> Gemini functionDeclarations. Only the documented
-- Schema subset survives; unknown keywords (strict, $schema,
-- additionalProperties, ...) are stripped because Gemini rejects them.
local SCHEMA_KEYS = {
  type = true, format = true, title = true, description = true,
  nullable = true, enum = true, properties = true, required = true,
  propertyOrdering = true, items = true, anyOf = true,
  minItems = true, maxItems = true, minLength = true, maxLength = true,
  pattern = true, minimum = true, maximum = true,
  minProperties = true, maxProperties = true, example = true, default = true,
}

local function sanitize_schema(s)
  if type(s) ~= "table" then return s end
  local out = {}
  for k, v in pairs(s) do
    if SCHEMA_KEYS[k] then
      if k == "properties" and type(v) == "table" then
        local props = {}
        for pk, pv in pairs(v) do props[pk] = sanitize_schema(pv) end
        out[k] = props
      elseif k == "items" then
        out[k] = sanitize_schema(v)
      elseif k == "anyOf" and type(v) == "table" then
        local anys = {}
        for _, sub in ipairs(v) do table.insert(anys, sanitize_schema(sub)) end
        out[k] = anys
      elseif k == "required" then
        -- Skip empty required array to avoid Gemini API validation error
        if type(v) == "table" and next(v) ~= nil then
          out[k] = v
        end
      else
        out[k] = v
      end
    end
  end
  return out
end

local function schema_nonempty(s)
  return type(s) == "table" and next(s) ~= nil
end

-- Built-in Gemini tools selectable via provider_config.google_tools, e.g.
-- {"googleSearch", "codeExecution"}. No OpenAI equivalent exists, so these
-- are static per-provider opt-ins, not per-request parameters.
local BUILTIN_TOOLS = {
  googleSearch = true, codeExecution = true, urlContext = true,
}

-- A request-level function tool named web_search (or google_search) is the
-- OpenAI-shaped spelling of the native googleSearch tool: there is no
-- function to execute client-side, the model searches on Google's servers.
local SEARCH_TOOL_NAMES = { web_search = true, google_search = true }

local function build_tools(request, provider_config)
  local tool_list = {}
  local builtin_set = {}
  local function enable_builtin(name)
    if not builtin_set[name] then
      builtin_set[name] = true
      table.insert(tool_list, { [name] = {} })
    end
  end
  local tools = request.tools
  if type(tools) == "table" and #tools > 0 then
    local decls = {}
    for _, t in ipairs(tools) do
      local fn = t["function"]
      local name, desc, params
      if type(fn) == "table" then
        name, desc, params = fn.name, fn.description, fn.parameters
      elseif type(t.name) == "string" then
        name, desc, params = t.name, t.description, t.parameters or t.input_schema
      end
      if type(name) ~= "string" or name == "" then
        return nil, nil, invalid_request("tool is missing a function name")
      end
      if SEARCH_TOOL_NAMES[name] then
        -- Native web search, not a client-executed function.
        enable_builtin("googleSearch")
      else
        local decl = { name = name, description = (desc ~= "" and desc or name) }
        if schema_nonempty(params) then decl.parameters = sanitize_schema(params) end
        table.insert(decls, decl)
      end
    end
    if #decls > 0 then
      table.insert(tool_list, 1, { functionDeclarations = decls })
    end
  end
  if type(provider_config) == "table" and type(provider_config.google_tools) == "table" then
    for _, name in ipairs(provider_config.google_tools) do
      if not BUILTIN_TOOLS[name] then
        return nil, nil, invalid_request("unknown built-in tool " .. tostring(name))
      end
      enable_builtin(name)
    end
  end
  local out_tools = nil
  if #tool_list > 0 then out_tools = tool_list end
  local tc = request.tool_choice
  local mode, allowed = nil, nil
  if tc == nil then
    -- omit: Gemini defaults to AUTO
  elseif tc == "none" then mode = "NONE"
  elseif tc == "auto" then mode = "AUTO"
  elseif tc == "required" then mode = "ANY"
  elseif type(tc) == "table" then
    if tc.type == "none" then mode = "NONE"
    elseif tc.type == "auto" then mode = "AUTO"
    elseif tc.type == "required" then mode = "ANY"
    elseif tc.type == "function" and type(tc["function"]) == "table"
        and tc["function"].name and tc["function"].name ~= "" then
      mode = "ANY"
      allowed = { tc["function"].name }
    end
  end
  local tool_config = nil
  if mode then
    tool_config = { functionCallingConfig = { mode = mode } }
    if allowed then tool_config.functionCallingConfig.allowedFunctionNames = allowed end
  end
  return out_tools, tool_config, nil
end

local function is_gemini3(model)
  return (model:lower():match("^gemini%-3") ~= nil)
end

local function request_model_name(model)
  local name = model:match("([^/]+)$") or model
  return (name:gsub("^models/", ""))
end

local function build_generation_config(request, model)
  local cfg = nil
  local function ensure()
    if not cfg then cfg = {} end
    return cfg
  end
  local max_tokens = request.max_completion_tokens or request.max_tokens
  if type(max_tokens) == "number" and max_tokens > 0 then
    ensure().maxOutputTokens = math.floor(max_tokens)
  end
  if request.temperature and request.temperature > 0 then
    ensure().temperature = request.temperature
  end
  if request.top_p and request.top_p > 0 then
    ensure().topP = request.top_p
  end
  if type(request.seed) == "number" and request.seed == math.floor(request.seed)
      and request.seed >= -2147483648 and request.seed <= 2147483647 then
    ensure().seed = request.seed
  end
  if request.stop ~= nil then
    local stops = {}
    if type(request.stop) == "string" then
      if request.stop ~= "" then stops = { request.stop } end
    elseif type(request.stop) == "table" then
      for _, s in ipairs(request.stop) do
        if (type(s) == "string" and s ~= "") or type(s) == "number" then
          table.insert(stops, tostring(s))
        end
        if #stops >= 5 then break end
      end
    end
    if #stops > 0 then ensure().stopSequences = stops end
  end
  -- NOTE: frequencyPenalty/presencePenalty/logprobs are intentionally not
  -- forwarded: Gemini rejects them on several models (e.g. 2.5-flash returns
  -- 400 "Penalty is not enabled" / "Logprobs is not enabled") and /models
  -- exposes no per-model capability flag for them.
  local effort = request.reasoning_effort
  if type(effort) == "string" and effort ~= "" then
    if effort == "none" then
      ensure().thinkingConfig = { thinkingBudget = 0 }
    elseif effort == "minimal" then
      -- Dynamic thinking, no summaries. thinkingLevel is Gemini-3-only.
      if is_gemini3(model) then ensure().thinkingConfig = { thinkingLevel = "MINIMAL" } end
    elseif effort == "default" then
      -- omit: server-side dynamic thinking
    else
      local level = effort:upper()
      if level == "XHIGH" or level == "MAX" then level = "HIGH" end
      if level ~= "LOW" and level ~= "MEDIUM" and level ~= "HIGH" then
        return nil, invalid_request("unknown reasoning_effort " .. tostring(effort))
      end
      if is_gemini3(model) then
        ensure().thinkingConfig = { thinkingLevel = level, includeThoughts = true }
      else
        ensure().thinkingConfig = { includeThoughts = true }
      end
    end
  end
  local rf = request.response_format
  if type(rf) == "table" and type(rf.type) == "string" then
    if rf.type == "text" then
      -- omit: text/plain is the default
    elseif rf.type == "json_object" then
      ensure().responseMimeType = "application/json"
    elseif rf.type == "json_schema" then
      ensure().responseMimeType = "application/json"
      if type(rf.json_schema) == "table" and schema_nonempty(rf.json_schema.schema) then
        ensure().responseSchema = sanitize_schema(rf.json_schema.schema)
      end
    else
      return nil, invalid_request("unsupported response_format type " .. rf.type)
    end
  end
  return cfg, nil
end

local function build_payload(request, provider_config)
  if type(request.n) == "number" and request.n > 1 then
    return nil, invalid_request("multiple candidates (n > 1) are not supported")
  end
  local model = request_model_name(request.model)
  local contents, system_parts, cerr = build_contents(request.messages)
  if cerr then return nil, cerr end
  local payload = { contents = contents }
  if #system_parts > 0 then
    payload.systemInstruction = { parts = system_parts }
  end
  local tools, tool_config, terr = build_tools(request, provider_config)
  if terr then return nil, terr end
  if tools then payload.tools = tools end
  if tool_config then payload.toolConfig = tool_config end
  local cfg, gerr = build_generation_config(request, model)
  if gerr then return nil, gerr end
  if cfg then payload.generationConfig = cfg end
  return payload, nil
end

-- Collect one candidate into text / reasoning / tool-call buckets shared by
-- complete and complete_stream. tool_seq numbers synthesized call ids so
-- streamed fragments of one call keep a stable id.
local function collect_parts(cand, out, tool_seq)
  local parts = cand and cand.content and cand.content.parts or {}
  for _, p in ipairs(parts) do
    if type(p) == "table" then
      if p.thought == true then
        if type(p.text) == "string" and p.text ~= "" then
          table.insert(out.reasoning, p.text)
        end
      elseif type(p.text) == "string" and p.text ~= "" then
        table.insert(out.text, p.text)
      elseif type(p.functionCall) == "table" then
        local fc = p.functionCall
        local args = "{}"
        if type(fc.args) == "table" and next(fc.args) ~= nil then
          args = json.encode(fc.args)
        end
        tool_seq.n = tool_seq.n + 1
        local call_id = (fc.id and fc.id ~= "") and fc.id or string.format("call_%d_%d", cand.index or 0, tool_seq.n)
        local call = {
          id = call_id,
          name = fc.name or "",
          arguments = args,
        }
        -- Preserve thought_signature for round-trip back to Gemini
        -- Workaround: encode into id since Crush doesn't preserve extra_content
        local sig = p.thoughtSignature or p.thought_signature
        if type(sig) == "string" and sig ~= "" then
          call.thought_signature = sig
          call.id = call_id .. "|" .. sig
        end
        table.insert(out.toolcalls, call)
      elseif type(p.inlineData) == "table" then
        -- Image/audio output has no OpenAI field; expose it as a data-URL
        -- markdown image so the bytes are not silently dropped.
        local mime = p.inlineData.mimeType or "application/octet-stream"
        local data = p.inlineData.data or ""
        if data ~= "" then
          table.insert(out.text, "![" .. mime .. "](data:" .. mime .. ";base64," .. data .. ")")
        end
      elseif type(p.fileData) == "table" and p.fileData.fileUri then
        table.insert(out.text, "[file](" .. tostring(p.fileData.fileUri) .. ")")
      elseif type(p.executableCode) == "table" then
        local code = p.executableCode.code or ""
        if code ~= "" then
          table.insert(out.text, "```" .. (p.executableCode.language or "") .. "\n" .. code .. "\n```")
        end
      elseif type(p.codeExecutionResult) == "table" then
        local output = p.codeExecutionResult.output or ""
        if output ~= "" then table.insert(out.text, output) end
      end
    end
  end
end

local function build_usage(meta)
  if not meta then return nil end
  local usage = {
    prompt_tokens = meta.promptTokenCount or 0,
    completion_tokens = meta.candidatesTokenCount or 0,
    total_tokens = meta.totalTokenCount or 0,
  }
  if meta.thoughtsTokenCount and meta.thoughtsTokenCount > 0 then
    usage.completion_tokens_details = { reasoning_tokens = meta.thoughtsTokenCount }
  end
  local function modality_sum(details, want)
    local n = 0
    for _, d in ipairs(details or {}) do
      if d.modality == want and type(d.tokenCount) == "number" then n = n + d.tokenCount end
    end
    return n
  end
  local prompt_details = nil
  if meta.cachedContentTokenCount and meta.cachedContentTokenCount > 0 then
    prompt_details = { cached_tokens = meta.cachedContentTokenCount }
  end
  local prompt_audio = modality_sum(meta.promptTokensDetails, "AUDIO")
  if prompt_audio > 0 then
    prompt_details = prompt_details or {}
    prompt_details.audio_tokens = prompt_audio
  end
  if prompt_details then usage.prompt_tokens_details = prompt_details end
  local completion_audio = modality_sum(meta.candidatesTokensDetails, "AUDIO")
  if completion_audio > 0 then
    usage.completion_tokens_details = usage.completion_tokens_details or {}
    usage.completion_tokens_details.audio_tokens = completion_audio
  end
  return usage
end

local function estimate_rpm(name)
  local m = name:lower()
  if m:find("flash") then
    if m:find("lite") then return 2000 end
    return 1000
  end
  if m:find("pro") then return 500 end
  return 1000
end

local function estimate_tpm(name)
  local m = name:lower()
  if m:find("flash") then
    if m:find("lite") then return 2000000 end
    return 1000000
  end
  if m:find("pro") then return 500000 end
  return 1000000
end

local function estimate_rpd(name)
  local m = name:lower()
  if m:find("flash") then
    if m:find("lite") then return 3000 end
    return 1500
  end
  if m:find("pro") then return 1000 end
  return 1500
end

local function model_short_name(entry)
  if type(entry.name) == "string" and entry.name ~= "" then
    return entry.name:gsub("^models/", "")
  end
  if type(entry.baseModelId) == "string" and entry.baseModelId ~= "" then
    return entry.baseModelId
  end
  return "unknown-model"
end

local function supports_generate_content(entry)
  for _, m in ipairs(entry.supportedGenerationMethods or {}) do
    if m == "generateContent" then return true end
  end
  return false
end

-- TTS preview models speak PCM only; image models (Nano Banana family) also
-- chat, so they keep chat/completions next to images/generations.
local function is_tts_model(name)
  return name:lower():find("tts", 1, true) ~= nil
end

local function is_image_model(name)
  local lower = name:lower()
  return lower:find("image", 1, true) ~= nil or lower:find("nano%-banana") ~= nil
end

-- RIFF/WAVE header around PCM s16le (the only encoding Gemini TTS emits).
local function wrap_wav(pcm, rate, channels)
  local bits = 16
  local function u32(n)
    return string.char(n % 256, math.floor(n / 256) % 256, math.floor(n / 65536) % 256, math.floor(n / 16777216) % 256)
  end
  local function u16(n)
    return string.char(n % 256, math.floor(n / 256) % 256)
  end
  local byte_rate = rate * channels * (bits / 8)
  local block_align = channels * (bits / 8)
  local data_size = #pcm
  return "RIFF" .. u32(36 + data_size) .. "WAVE"
    .. "fmt " .. u32(16) .. u16(1) .. u16(channels) .. u32(rate)
    .. u32(byte_rate) .. u16(block_align) .. u16(bits)
    .. "data" .. u32(data_size) .. pcm
end

-- Map an OpenAI size string ("1024x1024") to a Gemini aspect ratio.
local function aspect_ratio_of(size)
  if type(size) ~= "string" then return nil end
  local w, h = size:match("^(%d+)x(%d+)$")
  w, h = tonumber(w), tonumber(h)
  if not w or not h or w <= 0 or h <= 0 then return nil end
  local r = w / h
  local candidates = {
    { ratio = 1.0, label = "1:1" },
    { ratio = 4 / 3, label = "4:3" },
    { ratio = 3 / 4, label = "3:4" },
    { ratio = 3 / 2, label = "3:2" },
    { ratio = 2 / 3, label = "2:3" },
    { ratio = 16 / 9, label = "16:9" },
    { ratio = 9 / 16, label = "9:16" },
  }
  local best, best_diff = nil, math.huge
  for _, c in ipairs(candidates) do
    local diff = math.abs(c.ratio - r)
    if diff < best_diff then best, best_diff = c.label, diff end
  end
  return best
end

-- First candidate's inlineData part, or nil. Gemini returns generated
-- media as inlineData parts on the first candidate.
local function find_inline_data(g, family_prefix)
  if not g.candidates then return nil end
  for _, cand in ipairs(g.candidates) do
    local parts = cand and cand.content and cand.content.parts or {}
    for _, p in ipairs(parts) do
      local inline = type(p) == "table" and p.inlineData or nil
      if inline and type(inline.data) == "string" and inline.data ~= ""
          and type(inline.mimeType) == "string"
          and inline.mimeType:sub(1, #family_prefix) == family_prefix then
        return inline
      end
    end
  end
  return nil
end

local function blocked_error(g)
  if g and g.promptFeedback and g.promptFeedback.blockReason then
    return nil, { type = "invalid_request", message = "prompt blocked: " .. tostring(g.promptFeedback.blockReason) }
  end
  return nil, { type = "upstream", message = "upstream returned no generated content" }
end

llm_router.register("google", {
  icon = "https://www.gstatic.com/lamda/images/favicon_v1_150160cddff7f294ce30.svg",

  credential_schema = function()
    return {
      { type = "section", title = "Google AI Studio",
        subtitle = "Get your API key from Google AI Studio.",
        content = {
          { type = "link", text = "Open AI Studio", url = "https://aistudio.google.com/app/apikey" },
          { type = "secret", name = "api_key", label = "API Key", required = true },
          { type = "button", text = "Save", form_action = "submit" },
        } },
    }
  end,

  validate_credentials = function(data)
    local key = data.api_key
    if key == nil or key == "" then
      return false, { type = "invalid_request", message = "api_key is required" }
    end
    if #key < 20 then
      return false, { type = "invalid_request", message = "api_key: minimum 20 characters" }
    end
    return true
  end,

  get_model_infos = function(ctx, credential, provider_config)
    local client = llm_router.create_http_client({})
    local key = api_key_of(credential)
    local url = BASE_URL .. "/models"
    local infos = {}
    while url do
      local resp, err = client:request({
        method = "GET", url = url,
        headers = { ["x-goog-api-key"] = key },
      })
      if err then return nil, err end
      if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
      local page = json.decode(resp.body)
      for _, entry in ipairs(page.models or {}) do
        local name = model_short_name(entry)
        local skip = false
        if not supports_generate_content(entry) then skip = true end
        local lower = name:lower()
        -- Embedding models serve :embedContent instead of :generateContent;
        -- they are listed below with the embeddings endpoint, not skipped.
        local is_embedding = lower:find("embedding", 1, true) ~= nil
        if is_embedding then
          for _, m in ipairs(entry.supportedGenerationMethods or {}) do
            if m == "embedContent" then skip = false end
          end
        end
        if lower:find("aqa", 1, true) then skip = true end
        if not skip then
          local display = entry.displayName
          if display == nil or display == "" then display = name end
          -- Gemini vision/audio input is real for gemini-family models;
          -- gemma-4 takes images (audio unverified). Other families stay
          -- text-only. Output stays text: this adapter never requests
          -- IMAGE/AUDIO response modalities on chat completions.
          local in_modalities = { "text" }
          if lower:match("^gemini") then
            in_modalities = { "text", "image", "audio" }
          elseif lower:match("^gemma") then
            in_modalities = { "text", "image" }
          end
          local out_modalities = { "text" }
          local endpoints = nil
          local params = {
            "tools", "tool_choice", "response_format", "structured_outputs",
            "temperature", "top_p", "max_tokens", "seed", "stop",
            "reasoning", "reasoning_effort", "web_search",
          }
          if is_embedding then
            -- Embedding models take text in and return vectors; no chat.
            in_modalities = { "text" }
            out_modalities = { "embedding" }
            endpoints = { "embeddings" }
            params = { "dimensions" }
          elseif is_tts_model(name) then
            -- TTS previews take text in and speak audio out; they do not chat.
            in_modalities = { "text" }
            out_modalities = { "audio" }
            endpoints = { "audio/speech" }
            params = { "temperature" }
          elseif is_image_model(name) then
            -- Nano Banana family chats and renders images.
            out_modalities = { "text", "image" }
            endpoints = { "chat/completions", "images/generations" }
          end
          local info = {
            name = name, display_name = display,
            rpm = estimate_rpm(name), tpm = estimate_tpm(name), rpd = estimate_rpd(name),
            context_window = entry.inputTokenLimit or 0,
            max_tokens = entry.outputTokenLimit or 0,
            input_modalities = in_modalities,
            output_modalities = out_modalities,
            supported_parameters = params,
          }
          if endpoints then info.endpoints = endpoints end
          table.insert(infos, info)
          if entry.thinking == true then
            infos[#infos].reasoning = {
              default_enabled = true,
              supported_efforts = { "high", "medium", "low", "minimal", "none" },
            }
          end
        end
      end
      if page.nextPageToken and page.nextPageToken ~= "" then
        url = BASE_URL .. "/models?pageToken=" .. page.nextPageToken
      else
        url = nil
      end
    end
    return infos
  end,

  complete = function(ctx, credential, request)
    local blocked_err = quota_fail_fast(credential, request.model)
    if blocked_err then return nil, blocked_err end
    local client = llm_router.create_http_client({})
    local payload, perr = build_payload(request, ctx.provider_config)
    if perr then return nil, perr end
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/models/" .. request_model_name(request.model) .. ":generateContent",
      headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(payload),
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body, credential, request.model) end
    local g = json.decode(resp.body)
    if (not g.candidates or #g.candidates == 0) and g.promptFeedback and g.promptFeedback.blockReason then
      return nil, { type = "invalid_request", message = "prompt blocked: " .. tostring(g.promptFeedback.blockReason) }
    end
    local choices = {}
    if g.candidates and #g.candidates > 0 then
      local cand = g.candidates[1]
      local out = { text = {}, reasoning = {}, toolcalls = {} }
      collect_parts(cand, out, { n = 0 })
      local message = { role = "assistant", content = table.concat(out.text, "") }
      if #out.reasoning > 0 then
        message.reasoning_content = table.concat(out.reasoning, "")
      end
      if #out.toolcalls > 0 then
        local calls = {}
        for _, tc in ipairs(out.toolcalls) do
          local call = {
            id = tc.id, type = "function",
            ["function"] = { name = tc.name, arguments = tc.arguments },
          }
          -- Preserve thought_signature for round-trip back to Gemini
          if tc.thought_signature then
            if call.extra_content == nil then call.extra_content = {} end
            if call.extra_content.google == nil then call.extra_content.google = {} end
            call.extra_content.google.thought_signature = tc.thought_signature
          end
          table.insert(calls, call)
        end
        message.tool_calls = calls
      end
      local finish = map_finish_reason(cand.finishReason or "")
      if #out.toolcalls > 0 then finish = "tool_calls" end
      table.insert(choices, {
        index = cand.index or 0,
        message = message,
        finish_reason = finish,
      })
    end
    return {
      id = (g.responseId and g.responseId ~= "") and g.responseId or ("google-" .. tostring(os.time())),
      object = "chat.completion", created = os.time(),
      model = request.model, choices = choices,
      usage = build_usage(g.usageMetadata) or { prompt_tokens = 0, completion_tokens = 0, total_tokens = 0 },
    }
  end,

  complete_stream = function(ctx, credential, request, emit)
    local blocked_err = quota_fail_fast(credential, request.model)
    if blocked_err then return nil, blocked_err end
    local client = llm_router.create_http_client({})
    local payload, perr = build_payload(request, ctx.provider_config)
    if perr then return nil, perr end
    local full_model = request.model
    local chunk_id = "google-" .. tostring(os.time())
    local tool_seq = { n = 0 }
    -- Once any tool call is emitted, every later finish in this stream
    -- means "run the tools": Gemini sends a bare STOP after the call.
    local saw_tools = false
    local stream_err = client:stream({
      method = "POST",
      url = BASE_URL .. "/models/" .. request_model_name(request.model) .. ":streamGenerateContent?alt=sse",
      headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(payload),
      on_line = function(line)
        if line:sub(1, 6) ~= "data: " then return end
        local data = line:sub(7)
        if data == "[DONE]" then return end
        local ok, g = pcall(json.decode, data)
        if not ok or type(g) ~= "table" then return end
        local cand = g.candidates and g.candidates[1]
        if cand then
          local out = { text = {}, reasoning = {}, toolcalls = {} }
          collect_parts(cand, out, tool_seq)
          local delta = {}
          if #out.reasoning > 0 then delta.reasoning_content = table.concat(out.reasoning, "") end
          if #out.text > 0 then delta.content = table.concat(out.text, "") end
          if #out.toolcalls > 0 then
            local calls = {}
            for i, tc in ipairs(out.toolcalls) do
              table.insert(calls, {
                index = i - 1, id = tc.id, type = "function",
                ["function"] = { name = tc.name, arguments = tc.arguments },
              })
            end
            delta.tool_calls = calls
          end
          local choice = { index = cand.index or 0, delta = delta }
          if cand.finishReason and cand.finishReason ~= "" then
            choice.finish_reason = map_finish_reason(cand.finishReason)
          end
          if #out.toolcalls > 0 then
            saw_tools = true
            choice.finish_reason = "tool_calls"
          elseif saw_tools and choice.finish_reason then
            choice.finish_reason = "tool_calls"
          end
          if delta.content or delta.reasoning_content or delta.tool_calls or choice.finish_reason then
            emit({
              id = chunk_id, object = "chat.completion.chunk", created = os.time(),
              model = full_model, choices = { choice },
            })
          end
        end
        local usage = build_usage(g.usageMetadata)
        if usage then
          emit({
            id = chunk_id, object = "chat.completion.chunk", created = os.time(),
            model = full_model, choices = {}, usage = usage,
          })
        end
      end,
    })
    if stream_err then
      return classify_stream_error(stream_err, credential, request.model)
    end
  end,

  speech = function(ctx, credential, request)
    if type(request.input) ~= "string" or request.input == "" then
      return nil, invalid_request("input is required")
    end
    local voice = request.voice
    if type(voice) ~= "string" or voice == "" then voice = "Kore" end
    -- Style control on Gemini TTS is prompt-based: "Say cheerfully: ...".
    local text = request.input
    if type(request.instructions) == "string" and request.instructions ~= "" then
      text = request.instructions .. ": " .. text
    end
    local payload = {
      contents = { { role = "user", parts = { { text = text } } } },
      generationConfig = {
        responseModalities = { "AUDIO" },
        speechConfig = {
          voiceConfig = { prebuiltVoiceConfig = { voiceName = voice } },
        },
      },
    }
    local client = llm_router.create_http_client({})
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/models/" .. request_model_name(request.model) .. ":generateContent",
      headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode(payload),
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
    local g = json.decode(resp.body)
    local inline = find_inline_data(g, "audio/")
    if not inline then return blocked_error(g) end
    -- Upstream audio is always PCM s16le (mimeType "audio/L16;...;rate=24000").
    -- Only pcm and wav are servable without transcoding; every other
    -- requested format gets wav, the lossless playable container.
    local rate = tonumber(tostring(inline.mimeType):match("rate=(%d+)")) or 24000
    local requested = request.response_format
    if type(requested) ~= "string" or requested == "" then requested = "mp3" end
    if requested == "pcm" then
      return { audio_b64 = inline.data, format = "pcm" }
    end
    local pcm = llm_router.base64_decode(inline.data)
    local wav = wrap_wav(pcm, rate, 1)
    return { audio_b64 = llm_router.base64_encode(wav), format = "wav" }
  end,

  generate_image = function(ctx, credential, request)
    if type(request.prompt) ~= "string" or request.prompt == "" then
      return nil, invalid_request("prompt is required")
    end
    local n = 1
    if type(request.n) == "number" and request.n >= 1 then n = math.min(math.floor(request.n), 10) end
    local gen_config = { responseModalities = { "TEXT", "IMAGE" } }
    local ratio = aspect_ratio_of(request.size)
    if ratio then gen_config.imageConfig = { aspectRatio = ratio } end
    local payload = {
      contents = { { role = "user", parts = { { text = request.prompt } } } },
      generationConfig = gen_config,
    }
    local client = llm_router.create_http_client({})
    local data = {}
    -- One upstream call per requested image; image models do not take
    -- candidateCount.
    for _ = 1, n do
      local resp, err = client:request({
        method = "POST", url = BASE_URL .. "/models/" .. request_model_name(request.model) .. ":generateContent",
        headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
        body = json.encode(payload),
      })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
    local g = json.decode(resp.body)
      local inline = find_inline_data(g, "image/")
      if not inline then return blocked_error(g) end
      table.insert(data, { b64_json = inline.data })
    end
    return { created = os.time(), data = data }
  end,

  embed = function(ctx, credential, request)
    local input = request.input
    if type(input) ~= "table" or #input == 0 then
      return nil, invalid_request("input is required")
    end
    local model = request_model_name(request.model)
    local requests = {}
    for _, text in ipairs(input) do
      if type(text) ~= "string" or text == "" then
        return nil, invalid_request("input must not contain empty strings")
      end
      local entry = {
        model = "models/" .. model,
        content = { parts = { { text = text } } },
      }
      if type(request.dimensions) == "number" and request.dimensions > 0 then
        entry.outputDimensionality = math.floor(request.dimensions)
      end
      table.insert(requests, entry)
    end
    local client = llm_router.create_http_client({})
    local resp, err = client:request({
      method = "POST", url = BASE_URL .. "/models/" .. model .. ":batchEmbedContents",
      headers = { ["x-goog-api-key"] = api_key_of(credential), ["Content-Type"] = "application/json" },
      body = json.encode({ requests = requests }),
    })
    if err then return nil, err end
    if resp.status ~= 200 then return classify_error(resp.status, resp.headers, resp.body) end
    local g = json.decode(resp.body)
    local data = {}
    for i, item in ipairs(g.embeddings or {}) do
      if type(item.values) ~= "table" or #item.values == 0 then
        return nil, { type = "upstream", message = "upstream returned an empty embedding vector" }
      end
      table.insert(data, { index = i - 1, embedding = item.values })
    end
    if #data ~= #input then
      return nil, { type = "upstream", message = "upstream returned fewer embeddings than requested" }
    end
    return { data = data }
  end,
})
