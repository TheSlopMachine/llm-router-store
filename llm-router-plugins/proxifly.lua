--- @plugin Proxifly Proxy List
--- @author TheSlopMachine
--- @version 6.0.0
--- @plugin_api 1.0
--- @description Free proxy list from proxifly (proxies/all/data.json; http/https/socks4/socks5)
--- @allow_host cdn.jsdelivr.net

local LIST_URL = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/all/data.json"
local CACHE_SCOPE = "proxy_list"
local PROTOCOLS = { http = true, https = true, socks4 = true, socks5 = true }

llm_router.register_proxy_source("proxifly", {
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
      return nil, { message = "proxifly list: status " .. resp.status, code = "server_error", status = resp.status }
    end
    local ok, entries = pcall(json.decode, resp.body)
    if not ok or type(entries) ~= "table" then
      return nil, { message = "proxifly list: invalid json", code = "server_error", status = 502 }
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
    local out = {}
    for _, e in ipairs(entries) do
      if type(e) == "table" and PROTOCOLS[e.protocol] then
        if type(e.ip) == "string" and type(e.port) == "number" then
          local country = ""
          if type(e.geolocation) == "table" and type(e.geolocation.country) == "string" then
            country = e.geolocation.country
          end
          table.insert(out, { protocol = e.protocol, host = e.ip, port = e.port, country = country })
        end
      end
    end
    return out
  end,
})
