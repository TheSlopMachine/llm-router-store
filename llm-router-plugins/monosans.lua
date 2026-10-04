--- @plugin Monosans Proxy List
--- @author TheSlopMachine
--- @version 2.0.0
--- @router_version 0.7.0
--- @description Free HTTP proxy list from monosans (proxies/http.txt)
--- @allow_host cdn.jsdelivr.net

local LIST_URL = "https://cdn.jsdelivr.net/gh/monosans/proxy-list@main/proxies/http.txt"
local CACHE_SCOPE = "proxy_list"

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
        local host, port = line:match("^(.+):(%d+)$")
        port = tonumber(port)
        if type(host) == "string" and host ~= "" and type(port) == "number" and port >= 1 and port <= 65535 then
          table.insert(out, { protocol = "http", host = host, port = port, country = "" })
        end
      end
    end
    return out
  end,
})
