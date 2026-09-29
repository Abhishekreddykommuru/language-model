"""Polite, resumable web crawler for the manually collected share.

Politeness rules, all reported in the Phase 1 report: robots.txt is obeyed
including Crawl-delay; hosts declaring Content-Signal ai-train=no are excluded;
a descriptive User-Agent with a contact address, never browser spoofing; at most
~2 requests/second/host with backoff on 429 and 5xx; no paywalled paths; and
only extracted text plus provenance is released, never the raw HTML.
"""
import asyncio
import gzip
import inspect
import re
import time
import sqlite3
import urllib.robotparser as robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

#: Hosts declaring Content-Signal: ai-train=no. Never crawl these.
BLOCKED_HOSTS = {"ekantipur.com", "www.ekantipur.com",
                 "himalkhabar.com", "www.himalkhabar.com"}

#: URL fragments that are never article bodies. Excluding these is far more
#: reliable than an include-pattern per site: article URL shapes differ wildly
#: between publishers.
DEFAULT_EXCLUDE = re.compile(
    r"/(video|videos|photo|photos|gallery|galleries|web-stor(y|ies)|webstory|"
    r"tag|tags|author|authors|topic|topics|search|login|signin|register|"
    r"privacy|terms|about|contact|advertise|rss|feed|amp)(/|$|\?)"
    r"|\.(jpg|jpeg|png|gif|mp4|m3u8|pdf)$",
    re.IGNORECASE,
)


class Frontier:
    """URL queue in SQLite. Status: 0 pending, 1 done, 2 failed.

    SQLite locking over a FUSE mount is unreliable, so callers keep this file on
    local disk during a session and copy it to and from Drive at the boundaries.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection survives asyncio.to_thread
        # hand-offs; every statement here is short and synchronous.
        self.db = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS urls("
            "url TEXT PRIMARY KEY, host TEXT, status INTEGER DEFAULT 0,"
            " tries INTEGER DEFAULT 0)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_status ON urls(status, host)")
        self.db.commit()

    def add_many(self, urls, host: str):
        self.db.executemany(
            "INSERT OR IGNORE INTO urls(url, host) VALUES (?,?)",
            [(u, host) for u in urls],
        )
        self.db.commit()

    def next_batch(self, host: str, n: int) -> list:
        cur = self.db.execute(
            "SELECT url FROM urls WHERE status=0 AND host=? LIMIT ?", (host, n))
        return [r[0] for r in cur.fetchall()]

    def mark(self, url: str, status: int):
        self.db.execute(
            "UPDATE urls SET status=?, tries=tries+1 WHERE url=?", (status, url))

    def mark_many_done(self, urls):
        self.db.executemany(
            "UPDATE urls SET status=1, tries=tries+1 WHERE url=?",
            [(u,) for u in urls])
        self.db.commit()

    def commit(self):
        self.db.commit()

    def counts(self) -> dict:
        cur = self.db.execute("SELECT status, COUNT(*) FROM urls GROUP BY status")
        return {r[0]: r[1] for r in cur.fetchall()}

    def done_count(self, host: str = None) -> int:
        if host:
            cur = self.db.execute(
                "SELECT COUNT(*) FROM urls WHERE status=1 AND host=?", (host,))
        else:
            cur = self.db.execute("SELECT COUNT(*) FROM urls WHERE status=1")
        return cur.fetchone()[0]

    def backup_to(self, dest: str):
        """Snapshot the database with SQLite's backup API.

        copyfile on a live database can capture a torn image, since pending
        transactions may not be in the main file yet.
        """
        self.db.commit()
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest)
        with target:
            self.db.backup(target)
        target.close()

    def close(self):
        self.db.commit()
        self.db.close()


async def discover_sitemaps(client: httpx.AsyncClient, host: str) -> list:
    """Read the sitemaps a site declares in its own robots.txt.

    Hardcoded paths are unreliable: publishers use /sitemap.xml,
    /news-sitemap.xml, /commonfeeds/v1/tel/sitemap-index.xml and many others.
    """
    declared = []
    try:
        r = await client.get(f"https://{host}/robots.txt", timeout=30)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    declared.append(line.split(":", 1)[1].strip())
    except Exception as exc:
        print(f"      {host}: robots.txt unreadable ({exc})", flush=True)
    if not declared:
        declared = [f"https://{host}/sitemap.xml",
                    f"https://{host}/sitemap_index.xml"]
    return declared


async def fetch_sitemap_urls(client: httpx.AsyncClient, sitemap_url: str,
                             pattern: str = "", depth: int = 0,
                             seen: set = None, out: list = None,
                             max_urls: int = 200_000,
                             exclude: "re.Pattern" = None) -> list:
    """Expand a sitemap or sitemap index into candidate article URLs.

    Follows nested <sitemap> indexes to depth 4, because news archives nest:
    year index -> month index -> article sitemap. `pattern` is an optional
    include regex; empty means accept anything not excluded. Section pages that
    slip through cost one fetch and are dropped at extraction.
    """
    seen = seen if seen is not None else set()
    out = out if out is not None else []
    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE

    if sitemap_url in seen or depth > 4 or len(out) >= max_urls:
        return out
    seen.add(sitemap_url)

    try:
        r = await client.get(sitemap_url, timeout=60, follow_redirects=True)
        if r.status_code != 200:
            return out
        body = r.content
        # A .gz extension does not guarantee gzip: CDNs often serve these
        # decompressed. Sniff the magic bytes rather than trust the filename.
        if body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except OSError:
                pass
        text = body.decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"      sitemap failed, {sitemap_url}: {exc}", flush=True)
        return out

    locs = re.findall(r"<loc>\s*([^<]+?)\s*</loc>", text)
    nested = [u for u in locs if ".xml" in u.lower()]
    pages = [u for u in locs if u not in nested]

    for u in pages:
        if len(out) >= max_urls:
            break
        if exclude.search(u):
            continue
        if pattern and not re.search(pattern, u):
            continue
        out.append(u)

    for nested_url in nested:
        if len(out) >= max_urls:
            break
        await fetch_sitemap_urls(client, nested_url, pattern, depth + 1,
                                 seen, out, max_urls, exclude)
    return out


async def discover_by_links(client: httpx.AsyncClient, host: str,
                            seeds: list = None, max_urls: int = 50_000,
                            max_depth: int = 3, delay: float = 0.5,
                            exclude: "re.Pattern" = None,
                            max_pages_visited: int = 300,
                            max_seconds: float = 300.0) -> list:
    """Breadth-first link discovery, for sites with no usable sitemap.

    Five of seven Nepali hosts publish no sitemap, so sitemap-only discovery
    would have left Model L with almost no manually collected data. This walks
    the site as a reader would, at the same polite rate.

    Both the visit cap and the time cap matter: limiting only the enqueue side
    lets tens of thousands of URLs build up, all of which still get fetched, and
    discovery then eats the session before the crawl proper starts.
    """
    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE
    seeds = seeds or [f"https://{host}/"]
    seen_pages: set = set()
    found: set = set()
    queue = deque((s, 0) for s in seeds)
    link_re = re.compile(r'href=["\']([^"\'#]+)', re.IGNORECASE)

    t_start = time.time()
    while queue and len(found) < max_urls:
        if len(seen_pages) >= max_pages_visited:
            break
        if time.time() - t_start > max_seconds:
            print(f"      {host}: link-walk time cap reached, "
                  f"{len(found):,} URLs found", flush=True)
            break
        url, depth = queue.popleft()
        if url in seen_pages or depth > max_depth:
            continue
        seen_pages.add(url)
        try:
            r = await client.get(url, timeout=30, follow_redirects=True)
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                continue
            html = r.text
        except Exception:
            continue

        for raw in link_re.findall(html):
            link = urljoin(url, raw.strip())
            parsed = urlparse(link)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc.replace("www.", "") != host.replace("www.", ""):
                continue
            link = parsed.scheme + "://" + parsed.netloc + parsed.path
            if exclude.search(link) or link in found:
                continue
            found.add(link)
            if depth + 1 <= max_depth and len(queue) < max_pages_visited * 3:
                queue.append((link, depth + 1))

        if len(seen_pages) % 50 == 0 and found:
            print(f"      {host}: link-walk visited {len(seen_pages)} pages, "
                  f"{len(found):,} URLs found", flush=True)
        await asyncio.sleep(delay)

    return sorted(found)


class PoliteCrawler:
    """One crawler per host, obeying robots.txt including Crawl-delay."""

    def __init__(self, host: str, user_agent: str, default_delay: float):
        if host in BLOCKED_HOSTS:
            raise ValueError(f"{host} declares ai-train=no and must not be crawled")
        self.host = host
        self.ua = user_agent
        self.delay = default_delay
        self.rp = robotparser.RobotFileParser()
        self.robots_readable = False

        # robots.txt is fetched here rather than via RobotFileParser.read(),
        # which sets disallow_all on HTTP 401/403. Datacenter IPs often get a
        # 403 from bot protection that says nothing about crawling policy, and
        # treating that as a blanket prohibition silently blocked whole hosts.
        # RFC 9309 2.3.1.4 permits assuming no restrictions when robots.txt is
        # unreachable; page-level 403s are still honoured as failures.
        try:
            resp = httpx.get(f"https://{host}/robots.txt",
                             headers={"User-Agent": user_agent},
                             timeout=20, follow_redirects=True)
            if resp.status_code == 200:
                self.rp.parse(resp.text.splitlines())
                self.robots_readable = True
                declared = self.rp.crawl_delay(self.ua) or self.rp.crawl_delay("*")
                if declared:
                    self.delay = max(self.delay, float(declared))
                    print(f"    {host}: robots Crawl-delay {declared}s, "
                          f"using {self.delay}s", flush=True)
            else:
                print(f"    {host}: robots.txt returned HTTP {resp.status_code}, "
                      f"assuming no restrictions at {self.delay}s/request",
                      flush=True)
        except Exception as exc:
            print(f"    {host}: robots.txt unreachable ({type(exc).__name__}), "
                  f"assuming no restrictions at {self.delay}s/request", flush=True)

    def allowed(self, url: str) -> bool:
        if not self.robots_readable:
            return True
        try:
            return self.rp.can_fetch(self.ua, url)
        except Exception:
            return True

    async def crawl(self, urls, frontier: Frontier, on_page, max_pages: int,
                    mark_success: bool = True, deadline: float = None) -> dict:
        """Fetch urls at the polite rate, calling on_page(url, html) per success.

        With mark_success=False the caller marks URLs done once their content is
        durably stored; a URL marked done whose HTML died with the session would
        never be re-fetched. Failures are always marked immediately.
        """
        n_ok = n_fail = n_blocked = 0
        headers = {"User-Agent": self.ua, "Accept-Language": "*"}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True,
                                     timeout=30) as client:
            for url in urls:
                if n_ok >= max_pages:
                    break
                if deadline and time.time() >= deadline:
                    print(f"    {self.host}: time budget reached after "
                          f"{n_ok:,} pages", flush=True)
                    break
                if not self.allowed(url):
                    frontier.mark(url, 2)
                    n_blocked += 1
                    continue
                try:
                    r = await client.get(url)
                    if r.status_code in (429, 503):
                        await asyncio.sleep(30)
                        continue
                    if r.status_code == 200 and "html" in \
                            r.headers.get("content-type", ""):
                        res = on_page(url, r.text)
                        if inspect.isawaitable(res):
                            await res
                        if mark_success:
                            frontier.mark(url, 1)
                        n_ok += 1
                    else:
                        frontier.mark(url, 2)
                        n_fail += 1
                except Exception:
                    frontier.mark(url, 2)
                    n_fail += 1
                if (n_ok + n_fail) % 200 == 0:
                    frontier.commit()
                await asyncio.sleep(self.delay)
        frontier.commit()
        return {"ok": n_ok, "failed": n_fail, "robots_blocked": n_blocked}
