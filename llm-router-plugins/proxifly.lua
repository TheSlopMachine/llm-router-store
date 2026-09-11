--- @plugin Proxifly Proxy List
--- @author TheSlopMachine
--- @version 1.0.0
--- @router_version 0.0.4
--- @description Free proxy list from proxifly (proxies/all/data.json)
--- @allow_host raw.githubusercontent.com
--- @proxy_source true

local LIST_URL = "https://raw.githubusercontent.com/proxifly/free-proxy-list/refs/heads/main/proxies/all/data.json"

llm_router.register_proxy_source("proxifly", {
  fetch_proxies = function(ctx)
    local client = llm_router.create_http_client({ timeout_ms = 30000 })
    local resp, err = client:request({ method = "GET", url = LIST_URL })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, { type = "upstream", message = "proxifly list: status " .. resp.status }
    end
    local ok, entries = pcall(json.decode, resp.body)
    if not ok or type(entries) ~= "table" then
      return nil, { type = "upstream", message = "proxifly list: invalid json" }
    end
    local out = {}
    for _, e in ipairs(entries) do
      local proto = e.protocol
      if proto == "http" or proto == "https" or proto == "socks4" or proto == "socks5" then
        if type(e.ip) == "string" and type(e.port) == "number" then
          local country = ""
          if type(e.geolocation) == "table" and type(e.geolocation.country) == "string" then
            country = e.geolocation.country
          end
          table.insert(out, { protocol = proto, host = e.ip, port = e.port, country = country })
        end
      end
    end
    return out
  end,
})
