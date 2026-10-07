--- @plugin Monosans Proxy List
--- @author TheSlopMachine
--- @version 3.0.0
--- @plugin_api 1.0
--- @description Free proxy list from monosans (proxies/all.txt; http/socks4/socks5)
--- @allow_host cdn.jsdelivr.net

local LIST_URL = "https://cdn.jsdelivr.net/gh/monosans/proxy-list@main/proxies/all.txt"
local CACHE_SCOPE = "proxy_list"
local PROTOCOLS = { http = true, https = true, socks4 = true, socks4a = true, socks5 = true }

llm_router.register_proxy_source("monosans", {
  fetch_proxies = function()
    local client = llm_router.http_client({ timeout_ms = 30000 })
    local headers = {}
    local etag = llm_router.storage.get(CACHE_SCOPE, "etag")
    local last_modified = llm_router.storage.get(CACHE_SCOPE, "last_modified")
    if type(etag) == "string" then headers["If-None-Match"] = etag end
    if type(last_modified) == "string" then headers["If-Modified-Since"] = last_modified end

    local resp, err = client:request({ method = "GET", url = LIST_URL, headers = headers })
    if err then return nil, err end
    if resp.status == 304 then
      local response_headers = resp.headers or {}
      if type(response_headers.etag) == "string" then
        llm_router.storage.set(CACHE_SCOPE, "etag", response_headers.etag)
      end
      if type(response_headers["last-modified"]) == "string" then
        llm_router.storage.set(CACHE_SCOPE, "last_modified", response_headers["last-modified"])
      end
      return nil
    end
    if resp.status ~= 200 then
      return nil, { message = "monosans list: status " .. resp.status, code = "server_error", status = resp.status }
    end
    local response_headers = resp.headers or {}
    if type(response_headers.etag) == "string" then
      llm_router.storage.set(CACHE_SCOPE, "etag", response_headers.etag)
    else
      llm_router.storage.delete(CACHE_SCOPE, "etag")
    end
    if type(response_headers["last-modified"]) == "string" then
      llm_router.storage.set(CACHE_SCOPE, "last_modified", response_headers["last-modified"])
    else
      llm_router.storage.delete(CACHE_SCOPE, "last_modified")
    end
    local body = resp.body
    if type(body) ~= "string" or body == "" then
      return nil, { message = "monosans list: empty body", code = "server_error", status = 502 }
    end
    local out = {}
    for raw in (body .. "\n"):gmatch("([^\n]*)\n") do
      local line = (raw:gsub("\r", ""))
      line = line:match("^%s*(.-)%s*$") or ""
      if line ~= "" then
        local protocol, rest = line:match("^(%a[%w%+%-%.]*)://(.+)$")
        if rest == nil then
          protocol, rest = "http", line
        end
        if PROTOCOLS[protocol] then
          local host, port = rest:match("^(.+):(%d+)$")
          port = tonumber(port)
          if type(host) == "string" and host ~= "" and type(port) == "number" and port >= 1 and port <= 65535 then
            table.insert(out, { protocol = protocol, host = host, port = port, country = "" })
          end
        end
      end
    end
    return out
  end,
})
