--- @plugin Spys Proxy List
--- @author TheSlopMachine
--- @version 1.0.1
--- @plugin_api 1.4
--- @description Free proxy list from spys.me (proxy.txt http/https, socks.txt socks5)
--- @allow_host spys.me

local FEEDS = {
  { url = "https://spys.me/proxy.txt", kind = "http", scope = "spys_proxy" },
  { url = "https://spys.me/socks.txt", kind = "socks5", scope = "spys_socks" },
}

-- Unconditional full fetch once per interval. Validators persist durably,
-- so an emptied pool would otherwise 304 forever without reseeding.
local FULL_FETCH_TTL = 7 * 24 * 3600

llm_router.register_proxy_source("spys", {
  fetch_proxies = function()
    local client = llm_router.http_client({ timeout_ms = 30000 })
    local out = {}
    local first_err = nil
    for _, feed in ipairs(FEEDS) do
      local headers = {}
      -- Absent marker skips validators once, forcing a full list fetch.
      if llm_router.storage.get(feed.scope, "conditional_fetch") ~= nil then
        local etag = llm_router.storage.get(feed.scope, "etag")
        local last_modified = llm_router.storage.get(feed.scope, "last_modified")
        if type(etag) == "string" then headers["If-None-Match"] = etag end
        if type(last_modified) == "string" then headers["If-Modified-Since"] = last_modified end
      end

      local resp, err = client:request({ method = "GET", url = feed.url, headers = headers })
      if err then
        llm_router.log("warn", "feed request failed", { feed = feed.kind, reason = tostring(err.message or err) })
        if first_err == nil then first_err = err end
      elseif resp.status == 304 then
        llm_router.storage.set(feed.scope, "conditional_fetch", true, { ttl = FULL_FETCH_TTL })
        local response_headers = resp.headers or {}
        if type(response_headers.etag) == "string" then
          llm_router.storage.set(feed.scope, "etag", response_headers.etag)
        end
        if type(response_headers["last-modified"]) == "string" then
          llm_router.storage.set(feed.scope, "last_modified", response_headers["last-modified"])
        end
      elseif resp.status ~= 200 then
        llm_router.log("warn", "feed rejected", { feed = feed.kind, status = resp.status })
        if first_err == nil then
          first_err = { message = "spys list: status " .. resp.status, code = "server_error", status = resp.status }
        end
      else
        llm_router.storage.set(feed.scope, "conditional_fetch", true, { ttl = FULL_FETCH_TTL })
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
          llm_router.log("warn", "empty body", { feed = feed.kind })
          if first_err == nil then
            first_err = { message = "spys list: empty body", code = "server_error", status = 502 }
          end
        else
          for raw in (body .. "\n"):gmatch("([^\n]*)\n") do
            local line = (raw:gsub("\r", ""))
            line = line:match("^%s*(.-)%s*$") or ""
            -- Lines look like "77.37.209.86:1081 RU-H -": address, info,
            -- gap flag. No end anchor: the trailing gap flag follows info.
            local host, port, info = line:match("^([^%s:]+):(%d+)%s+(%S+)")
            port = tonumber(port)
            if type(host) == "string" and host ~= "" and type(port) == "number" and port >= 1 and port <= 65535 and type(info) == "string" then
              local country, flags = info:match("^([A-Z][A-Z])%-(.*)$")
              if type(country) == "string" then
                local protocol = feed.kind
                if feed.kind == "http" and type(flags) == "string" and flags:find("S", 1, true) then
                  protocol = "https"
                end
                table.insert(out, { protocol = protocol, host = host, port = port, country = country })
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
    llm_router.log("debug", "list served", { count = #out })
    return out
  end,
})
