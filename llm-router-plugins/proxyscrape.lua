--- @plugin Proxyscrape Proxy List
--- @author TheSlopMachine
--- @version 1.0.0
--- @plugin_api 1.4
--- @description Free proxy list from proxyscrape v2 API (http, socks4, socks5)
--- @allow_host api.proxyscrape.com

-- Only labeled feeds: protocol=all mixes protocols without labels, and the
-- ssl=yes/no splits are subsets of ssl=all.
local FEEDS = {
  { url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all", protocol = "http", scope = "proxyscrape_http" },
  { url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=socks4&timeout=10000&country=all", protocol = "socks4", scope = "proxyscrape_socks4" },
  { url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=socks5&timeout=10000&country=all", protocol = "socks5", scope = "proxyscrape_socks5" },
}

-- Unconditional full fetch once per interval. Validators persist durably,
-- so an emptied pool would otherwise 304 forever without reseeding.
local FULL_FETCH_TTL = 7 * 24 * 3600

llm_router.register_proxy_source("proxyscrape", {
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
        llm_router.log("warn", "feed request failed", { feed = feed.protocol, reason = tostring(err.message or err) })
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
        llm_router.log("warn", "feed rejected", { feed = feed.protocol, status = resp.status })
        if first_err == nil then
          first_err = { message = "proxyscrape list: status " .. resp.status, code = "server_error", status = resp.status }
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
          llm_router.log("warn", "empty body", { feed = feed.protocol })
          if first_err == nil then
            first_err = { message = "proxyscrape list: empty body", code = "server_error", status = 502 }
          end
        else
          -- Bare IP:Port lines, no metadata.
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
    llm_router.log("debug", "list served", { count = #out })
    return out
  end,
})
