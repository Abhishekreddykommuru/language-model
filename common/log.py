"""Console output helpers, so every stage reports in the same shape."""


def section(title: str):
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def info(msg: str, indent: int = 1):
    print("  " * indent + msg, flush=True)


def warn(msg: str):
    print(f"  ! {msg}", flush=True)


def kv(label: str, value, width: int = 26):
    print(f"  {label:<{width}} {value}", flush=True)


def table(headers: list, rows: list, aligns: str = None):
    """Print an aligned table. `aligns` is one of 'l'/'r' per column."""
    cols = [[str(h)] + [str(r[i]) for r in rows] for i, h in enumerate(headers)]
    widths = [max(len(c) for c in col) for col in cols]
    aligns = aligns or "l" + "r" * (len(headers) - 1)

    def line(cells):
        return "  " + "  ".join(
            c.rjust(w) if a == "r" else c.ljust(w)
            for c, w, a in zip(cells, widths, aligns))

    print(line([str(h) for h in headers]), flush=True)
    print("  " + "  ".join("-" * w for w in widths), flush=True)
    for r in rows:
        print(line([str(c) for c in r]), flush=True)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def millions(n: float) -> str:
    return f"{n / 1e6:,.1f}M"


def progress(iterable, desc: str, total: int = None, unit: str = "doc"):
    """Wrap an iterable in a live progress bar.

    Falls back to periodic prints where tqdm is unavailable, so a stage never
    fails for want of a progress bar. Most corpus stages stream generators of
    unknown length, so `total` is usually None and the bar shows count, rate
    and elapsed time rather than a percentage.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit=unit,
                    unit_scale=True, dynamic_ncols=True, smoothing=0.05,
                    miniters=1000)
    except ImportError:
        def _fallback():
            for i, item in enumerate(iterable, 1):
                if i % 100_000 == 0:
                    print(f"  {desc}: {i:,} {unit}s", flush=True)
                yield item
        return _fallback()


def count_shards(*dirs) -> int:
    """How many shards a stage will read, for a bar with a real total."""
    from pathlib import Path

    return sum(len(list(Path(d).rglob("*.jsonl.zst")))
               for d in dirs if Path(d).exists())
