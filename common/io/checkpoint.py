"""Resumable stage runner.

A Colab session dies at 12 hours, or sooner if the tab goes idle. Stages
process a list of work units and write a _SUCCESS marker per finished unit onto
Drive, so a lost session costs at most one unit.
"""
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable


class Checkpoint:
    """Tracks completed units of work in a directory on durable storage."""

    def __init__(self, state_dir: str):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _marker(self, unit: Any) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(unit))
        return self.dir / f"{safe[:180]}._SUCCESS"

    def done(self, unit: Any) -> bool:
        return self._marker(unit).exists()

    def mark(self, unit: Any, meta: dict = None):
        payload = {
            "unit": str(unit),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if meta:
            payload.update(meta)
        self._marker(unit).write_text(json.dumps(payload, ensure_ascii=False))

    def summary(self) -> dict:
        out = {}
        for p in self.dir.glob("*._SUCCESS"):
            try:
                out[p.name] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return out


def run_resumable(units: Iterable[Any], state_dir: str,
                  fn: Callable[[Any], dict], desc: str = "") -> dict:
    """Apply fn to each unit, skipping units already marked complete.

    Returns the numeric counters returned by fn, summed across this run.
    """
    ck = Checkpoint(state_dir)
    units = list(units)
    totals: dict = {}
    for i, unit in enumerate(units, 1):
        tag = f"[{desc} {i}/{len(units)}]"
        if ck.done(unit):
            print(f"{tag} skipping {unit}, already done", flush=True)
            continue
        print(f"{tag} {unit}", flush=True)
        meta = fn(unit) or {}
        ck.mark(unit, meta)
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0) + v
        print(f"{tag} finished {unit}: {meta}", flush=True)
    return totals
