--- @plugin Proxifly Proxy List
--- @author TheSlopMachine
--- @version 3.1.0
--- @router_version 0.3.0
--- @description Free proxy list from proxifly (proxies/all/data.json)
--- @allow_host raw.githubusercontent.com
--- @proxy_source true

local LIST_URL = "https://raw.githubusercontent.com/proxifly/free-proxy-list/refs/heads/main/proxies/all/data.json"
local CACHE_SCOPE = "proxy_list"

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
      return nil, { type = "upstream", message = "proxifly list: status " .. resp.status }
    end
    local ok, entries = pcall(json.decode, resp.body)
    if not ok or type(entries) ~= "table" then
      return nil, { type = "upstream", message = "proxifly list: invalid json" }
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
      if type(e) == "table" and e.protocol == "http" then
        if type(e.ip) == "string" and type(e.port) == "number" then
          local country = ""
          if type(e.geolocation) == "table" and type(e.geolocation.country) == "string" then
            country = e.geolocation.country
          end
          table.insert(out, { protocol = "http", host = e.ip, port = e.port, country = country })
        end
      end
    end
    return out
  end,
})
