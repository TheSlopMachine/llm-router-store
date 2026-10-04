--- @plugin TheSpeedX Proxy List
--- @author TheSlopMachine
--- @version 2.1.0
--- @router_version 0.7.0
--- @description Free proxy list from TheSpeedX (http.txt, socks4.txt, socks5.txt)
--- @allow_host cdn.jsdelivr.net

local FEEDS = {
  { url = "https://cdn.jsdelivr.net/gh/TheSpeedX/PROXY-List@master/http.txt", protocol = "http", scope = "proxy_list_http" },
  { url = "https://cdn.jsdelivr.net/gh/TheSpeedX/PROXY-List@master/socks4.txt", protocol = "socks4", scope = "proxy_list_socks4" },
  { url = "https://cdn.jsdelivr.net/gh/TheSpeedX/PROXY-List@master/socks5.txt", protocol = "socks5", scope = "proxy_list_socks5" },
}

llm_router.register_proxy_source("thespeedx", {
  fetch_proxies = function()
    local client = llm_router.http_client({ timeout_ms = 30000 })
    local out = {}
    local first_err = nil
    for _, feed in ipairs(FEEDS) do
      local headers = {}
      local etag = llm_router.storage.get(feed.scope, "etag")
      local last_modified = llm_router.storage.get(feed.scope, "last_modified")
      if type(etag) == "string" then headers["If-None-Match"] = etag end
      if type(last_modified) == "string" then headers["If-Modified-Since"] = last_modified end

      local resp, err = client:request({ method = "GET", url = feed.url, headers = headers })
      if err then
        if first_err == nil then first_err = err end
      elseif resp.status == 304 then
        local response_headers = resp.headers or {}
        if type(response_headers.etag) == "string" then
          llm_router.storage.set(feed.scope, "etag", response_headers.etag)
        end
        if type(response_headers["last-modified"]) == "string" then
          llm_router.storage.set(feed.scope, "last_modified", response_headers["last-modified"])
        end
      elseif resp.status ~= 200 then
        if first_err == nil then
          first_err = { message = "thespeedx list: status " .. resp.status, code = "server_error", status = resp.status }
        end
      else
        local response_headers = resp.headers or {}
        if type(response_headers.etag) == "string" then
          llm_router.storage.set(feed.scope, "etag", response_headers.etag)
        else
          llm_router.storage.delete(feed.scope, "etag")
        end
        if type(response_headers["last-modified"]) == "string" then
          llm_router.storage.set(feed.scope, "last_modified", response_headers["last-modified"])
        else
          llm_router.storage.delete(feed.scope, "last_modified")
        end
        local body = resp.body
        if type(body) ~= "string" or body == "" then
          if first_err == nil then
            first_err = { message = "thespeedx list: empty body", code = "server_error", status = 502 }
          end
        else
          for raw in (body .. "\n"):gmatch("([^\n]*)\n") do
            local line = (raw:gsub("\r", ""))
            line = line:match("^%s*(.-)%s*$") or ""
            if line ~= "" then
              local host, port = line:match("^(.+):(%d+)$")
              port = tonumber(port)
              if type(host) == "string" and host ~= "" and type(port) == "number" and port >= 1 and port <= 65535 then
                table.insert(out, { protocol = feed.protocol, host = host, port = port, country = "" })
              end
            end
          end
        end
      end
    end
    if #out == 0 then
      if first_err ~= nil then return nil, first_err end
      return nil
    end
    return out
  end,
})
