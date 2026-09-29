"""Stage 1: crawl our own pages, which is the manually collected share.

Resumable across sessions. The SQLite frontier is copied from Drive at the
start, worked on locally because FUSE file locking is unreliable, and copied
back as the crawl checkpoints. Raw HTML goes to local disk and is archived to
Drive in batches, so extraction can be re-run later with better rules without
re-crawling anything.

Usage:
    python scripts/02_crawl.py --lang tel --max-pages-per-host 60000
"""
import argparse
import asyncio
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, ".")

from common.collect.crawler import (BLOCKED_HOSTS, Frontier, PoliteCrawler,
                                    discover_by_links, discover_sitemaps,
                                    fetch_sitemap_urls)

#: Below this many sitemap URLs, fall back to walking links. Checked live: five
#: of seven Nepali hosts publish no usable sitemap at all.
SITEMAP_FALLBACK_THRESHOLD = 200
from common.io.paths import (ensure_drive_mounted, lang_dir, scratch_dir,
                             work_dir)
from common.log import kv, section

#: Pages per HTML archive written to Drive. This is the re-fetch window, not a
#: loss window: a URL is marked done only once its HTML is archived, so an
#: interrupted session re-fetches at most this many pages. Roughly 10 minutes.
ARCHIVE_EVERY = 2000

#: Archive on time as well as on count, so a slow host still checkpoints.
#: Ratopati declares Crawl-delay: 20.
ARCHIVE_MAX_SECONDS = 600

#: Floor for the time trigger. Without it a slow host emits an archive every few
#: pages, leaving 02b with tens of thousands of Drive round trips to make.
MIN_ARCHIVE_PAGES = 250


async def discover(site: dict, user_agent: str, frontier: Frontier,
                   max_urls: int = 200_000) -> int:
    """Expand a site's sitemaps into article URLs and seed the frontier.

    Sitemap locations come from the site's own robots.txt, plus anything listed
    in sources.yaml. Guessing conventional paths produced zero URLs for most
    hosts, so it is not attempted.
    """
    host = site["host"]
    async with httpx.AsyncClient(headers={"User-Agent": user_agent},
                                 follow_redirects=True, timeout=60) as client:
        sitemaps = await discover_sitemaps(client, host)
        for extra in site.get("sitemaps", []) or []:
            if extra not in sitemaps:
                sitemaps.append(extra)
        print(f"    {host}: {len(sitemaps)} sitemap(s) declared", flush=True)

        urls, seen = [], set()
        for sitemap in sitemaps:
            if len(urls) >= max_urls:
                break
            await fetch_sitemap_urls(
                client, sitemap, site.get("article_url_pattern", ""),
                seen=seen, out=urls, max_urls=max_urls)

        urls = sorted(set(u for u in urls if u.startswith("http")))

        # Fall back to link-walking when the sitemap route comes up short.
        if len(urls) < SITEMAP_FALLBACK_THRESHOLD:
            print(f"    {host}: only {len(urls)} URLs from sitemaps; "
                  f"falling back to link discovery", flush=True)
            seeds = [f"https://{host}/"] + (site.get("seed_pages") or [])
            linked = await discover_by_links(
                client, host, seeds=seeds, max_urls=max_urls,
                delay=site.get("delay", 0.5),
                max_pages_visited=300, max_seconds=300)
            urls = sorted(set(urls) | set(linked))

    frontier.add_many(urls, host)
    if not urls:
        print(f"    {host}: no URLs discovered, skipping this host. Usually the "
              f"site blocks datacenter IPs, or publishes no sitemap and renders "
              f"its homepage in JavaScript.", flush=True)
    else:
        print(f"    {host}: {len(urls):,} candidate article URLs", flush=True)
    return len(urls)


async def crawl_host(site: dict, cfg: dict, frontier, html_dir: Path,
                     archive_dir: Path, remote_db: Path, max_pages: int,
                     deadline: float = None) -> dict:
    """Crawl one host end to end. Hosts run concurrently; see main().

    Politeness is per host, so ten hosts at once is ten times the throughput
    while no single site sees more than its 2 requests/second. Sequentially, the
    whole pipeline ran at one host's rate and later hosts were never reached.
    """
    host = site["host"]
    if host in BLOCKED_HOSTS:
        print(f"    {host}: skipped, declares Content-Signal ai-train=no",
              flush=True)
        return {"host": host, "ok": 0, "skipped": True}

    if not frontier.next_batch(host, 1):
        await discover(site, cfg["user_agent"], frontier)

    crawler = PoliteCrawler(host, cfg["user_agent"], cfg["default_delay_seconds"])
    out = html_dir / host
    out.mkdir(parents=True, exist_ok=True)
    already = frontier.done_count(host)
    counter = {"n": 0, "batch": 0, "last": time.time()}
    pending: list = []

    def _tar(tar_path: str) -> int:
        return subprocess.run(f"tar -cf '{tar_path}' -C '{out}' .",
                              shell=True).returncode

    async def archive_batch():
        """Tar this batch to Drive, then mark its URLs done.

        The ordering is the point: marking a URL done before its content is
        durable leaves URLs recorded as fetched whose HTML died with the
        session, and they are never re-fetched.
        """
        files = sorted(out.glob("*.html.gz"))
        if not files:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        tar_path = f"{archive_dir}/html-{host}-{counter['batch']:04d}-{stamp}.tar"
        rc = await asyncio.to_thread(_tar, tar_path)

        # Update the clock before deciding success. Skipping it on the failure
        # path leaves the time trigger permanently true, so every subsequent
        # page fires another archive; that is how one run produced 67,180 tiny
        # archives instead of a few dozen.
        counter["last"] = time.time()

        # Drive is eventually consistent, so a file tar has just written may not
        # be visible through the mount yet. Give it a moment before giving up.
        ok = rc == 0
        if ok and not Path(tar_path).exists():
            await asyncio.sleep(2)
            ok = Path(tar_path).exists()
        if not ok:
            print(f"      {host}: archive failed (rc={rc}), URLs stay pending "
                  f"and retry next batch", flush=True)
            return
        for f in files:
            f.unlink(missing_ok=True)
        frontier.mark_many_done(pending)
        frontier.backup_to(str(remote_db))   # fast; keep on the loop thread
        counter["batch"] += 1
        print(f"      {host}: archived {len(files):,} pages and committed "
              f"{len(pending):,} URLs to {Path(tar_path).name}", flush=True)
        pending.clear()

    async def on_page(url: str, html: str):
        name = f"{abs(hash(url)) % (10 ** 12)}.html.gz"
        with gzip.open(out / name, "wt", encoding="utf-8") as fh:
            fh.write(url + "\n" + html)
        pending.append(url)
        counter["n"] += 1
        if counter["n"] % 500 == 0:
            print(f"      {host}: {counter['n']:,} this session, "
                  f"{already + counter['n']:,} total", flush=True)
        # The page floor stops the time trigger producing tiny archives on a
        # slow host. The final flush in `finally` has no floor, so nothing is
        # left unarchived.
        if (len(pending) >= ARCHIVE_EVERY
                or (len(pending) >= MIN_ARCHIVE_PAGES
                    and time.time() - counter["last"] > ARCHIVE_MAX_SECONDS)):
            await archive_batch()

    urls = frontier.next_batch(host, max_pages)
    print(f"    {host}: {already:,} fetched previously, {len(urls):,} pending",
          flush=True)
    if not urls:
        return {"host": host, "ok": 0, "note": "nothing pending"}

    try:
        result = await crawler.crawl(urls, frontier, on_page, max_pages,
                                     mark_success=False, deadline=deadline)
    finally:
        await archive_batch()       # also runs on Ctrl+C and teardown
    result["host"] = host
    print(f"    {host}: finished with {result['ok']:,} pages, "
          f"{result['failed']:,} failed, {result['robots_blocked']:,} "
          f"disallowed by robots", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--max-pages-per-host", type=int, default=40000)
    ap.add_argument("--html-dir", default=None)
    ap.add_argument("--sequential", action="store_true",
                    help="crawl hosts one at a time (slow; for debugging)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after this many hours, archiving and "
                         "committing the batch in flight. More useful than "
                         "guessing a page count against a 12 h session cap.")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    html_dir = Path(args.html_dir or f"{scratch_dir()}/raw_html/{args.lang}")
    html_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/sources.yaml",
                              encoding="utf-8"))["crawl"]

    remote_db = Path(f"{work}/_state/frontier_{args.lang}.sqlite")
    local_db = Path(f"{scratch_dir()}/frontier_{args.lang}.sqlite")
    remote_db.parent.mkdir(parents=True, exist_ok=True)
    section(f"Crawling {args.lang}")
    if remote_db.exists():
        kv("frontier", f"resumed from Drive, "
                       f"{remote_db.stat().st_size / 1e6:.1f} MB")
    elif local_db.exists():
        # A stale local frontier with none on Drive would silently treat a
        # previous run's URLs as already fetched.
        local_db.unlink()
        kv("frontier", "none on Drive, starting fresh")
    frontier = Frontier(str(local_db))

    archive_dir = Path(f"{work}/_raw_html/{args.lang}")
    archive_dir.mkdir(parents=True, exist_ok=True)

    deadline = (time.time() + args.max_hours * 3600) if args.max_hours else None
    if deadline:
        kv("time budget", f"{args.max_hours:.1f} h, then stops cleanly")

    async def run_all_hosts():
        tasks = [crawl_host(site, cfg, frontier, html_dir, archive_dir,
                            remote_db, args.max_pages_per_host, deadline)
                 for site in cfg["sites"]]
        if args.sequential:
            return [await t for t in tasks]
        # Concurrency across hosts is what makes ten hosts worth having: 20
        # requests/second overall, none of which any single site sees.
        kv("hosts", f"{len(tasks)}, crawled concurrently")
        print()
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_all_hosts())
    for r in results:
        if isinstance(r, Exception):
            print(f"    host failed, {type(r).__name__}: {r}", flush=True)

    counts = frontier.counts()
    frontier.backup_to(str(remote_db))
    frontier.close()

    section(f"Crawl finished, {args.lang}")
    kv("pages fetched", f"{counts.get(1, 0):,}")
    kv("URLs still pending", f"{counts.get(0, 0):,}")
    kv("URLs failed", f"{counts.get(2, 0):,}")
    kv("HTML archives on Drive", f"{len(list(archive_dir.glob('*.tar'))):,}")
    kv("archive location", archive_dir)


if __name__ == "__main__":
    main()
