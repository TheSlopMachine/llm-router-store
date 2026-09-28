--- @plugin Proxifly SOCKS5 Proxy List
--- @author TheSlopMachine
--- @version 3.0.0
--- @router_version 0.3.0
--- @description Free SOCKS5 proxy list from proxifly (proxies/countries/US/data.json)
--- @allow_host cdn.jsdelivr.net
--- @proxy_source true

local LIST_URL = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/countries/US/data.json"

llm_router.register_proxy_source("proxifly-socks5", {
  fetch_proxies = function()
    local client = llm_router.http_client({ timeout_ms = 30000 })
    local resp, err = client:request({ method = "GET", url = LIST_URL })
    if err then return nil, err end
    if resp.status ~= 200 then
      return nil, { type = "upstream", message = "proxifly-socks5 list: status " .. resp.status }
    end
    local ok, entries = pcall(json.decode, resp.body)
    if not ok or type(entries) ~= "table" then
      return nil, { type = "upstream", message = "proxifly-socks5 list: invalid json" }
    end
    local out = {}
    for _, e in ipairs(entries) do
      local proto = e.protocol
      if proto == "socks5" then
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
