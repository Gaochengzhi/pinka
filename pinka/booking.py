"""What a job is booked at: VRAM, cores and RAM from its declaration, its own measurements and the history of earlier
runs under the same history key. A declaration is an upper bound; what the same job measured is the estimate. A
declaration that proved too low stops being a cap. README.md, "How packing and booking work", has the rules in prose.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from .config import atomic_json, read_json

VRAM_MARGIN, VRAM_PAD_GB = 1.2, 1.0     # own booking = margin x measured peak + pad
HIST_VRAM_MARGIN = 1.1                  # history cap = margin x p95 of the key's peaks + pad
OVER_MARGIN, OVER_PAD_GB = 1.1, 1.0     # measured above margin x declared + pad: the declaration was too low
RAM_OVER_MARGIN, RAM_OVER_PAD_GB = 1.1, 2.0
CPU_MARGIN = 1.2
HIST_KEEP = 20
_SUFFIX = re.compile(r"(?:[-_.](?:[a-z]?\d+)|\d+)+$")


def hist_prefix(name: str) -> str:
    """History key of a name: lower case, trailing -s3 / -k2 / _t0 / -007 / digits stripped, so `eval-s3` and
    `eval-k12` share one history."""
    n = name.lower()
    return _SUFFIX.sub("", n) or n


def hist_key(spec) -> str:
    """A spec dict (its `key`, else its name) or a plain name."""
    if isinstance(spec, str):
        return hist_prefix(spec)
    return hist_prefix(spec.get("key") or spec["name"])


def _p95(xs: list) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]


def known_caps(hist: dict, spec, min_n: int = 3) -> dict:
    """{vram_gb, cpu, ram_gb} that finished runs under this key needed, with margin (VRAM: p95, so one spiky run does
    not cost every later job a slot; cores and RAM: the max); only keys with >= min_n samples."""
    e = (hist or {}).get(hist_key(spec)) or {}
    out = {}
    for key, src, scale, pad, agg in (("vram_gb", "vram", HIST_VRAM_MARGIN, VRAM_PAD_GB, _p95),
                                      ("cpu", "cores", CPU_MARGIN, 0.0, max), ("ram_gb", "ram", 1.2, 1.0, max)):
        xs = e.get(src) or []
        if len(xs) >= min_n:
            out[key] = round(agg(xs) * scale + pad, 1)
    return out


def hist_max(hist: dict, spec, key: str = "vram") -> float:
    """Largest recorded value under the key (one run is enough)."""
    return max(((hist or {}).get(hist_key(spec)) or {}).get(key) or [0.0])


def hist_record(path: Path, spec, vram_gb: float = 0.0, cores: float = 0.0, ram_gb: float = 0.0) -> None:
    h = read_json(path, {}) or {}
    e = h.setdefault(hist_key(spec), {})
    for k, v in (("vram", vram_gb), ("cores", cores), ("ram", ram_gb)):
        if v > 0:
            e[k] = (e.get(k, []) + [round(v, 2)])[-HIST_KEEP:]
    atomic_json(path, h)


def history_defaults(hist: dict, name: str) -> dict:
    """{vram_gb, cpu, ram_gb} = p95 x 1.2 of the runs under this name's key: what `submit` uses for an omitted flag."""
    e = (hist or {}).get(hist_prefix(name)) or {}
    out = {}
    if e.get("vram"):
        out["vram_gb"] = round(math.ceil(_p95(e["vram"]) * 1.2 * 10) / 10, 1)
    if e.get("cores"):
        out["cpu"] = max(1, math.ceil(_p95(e["cores"]) * 1.2))
    if e.get("ram"):
        out["ram_gb"] = round(math.ceil(_p95(e["ram"]) * 1.2 * 10) / 10, 1)
    return out


def is_gpu(spec: dict) -> bool:
    return float(spec.get("vram_gb") or 0) > 0 or bool(spec.get("exclusive"))


# ------------------------------------------------------------------------------------------------ VRAM
def over_declared(spec: dict, measured: float) -> bool:
    return not spec.get("exclusive") and measured > OVER_MARGIN * float(spec.get("vram_gb") or 0) + OVER_PAD_GB


def vram_raised(spec: dict, peak: float = 0.0, seen: float = 0.0) -> float:
    """GB to book for a job whose declaration proved too low (0 when it did not): 1.2 x its own peak + 1 (it may
    still grow), or the key's recorded peak `seen` + 1."""
    return round(max(VRAM_MARGIN * peak + VRAM_PAD_GB if over_declared(spec, peak) else 0.0,
                     seen + VRAM_PAD_GB if over_declared(spec, seen) else 0.0), 1)


def vram_need(spec: dict, known: dict, trust: bool = True, seen: float = 0.0) -> float:
    """GB a queued job needs free on a card."""
    up = vram_raised(spec, 0.0, seen)
    if up:
        return up
    declared = float(spec.get("vram_gb") or 0)
    if not trust or spec.get("exclusive") or "vram_gb" not in (known or {}):
        return declared
    return min(declared, known["vram_gb"])


def vram_charge(j: dict, now: float, known: dict = None, trust: bool = True, seen: float = 0.0,
                settle_s: float = 600.0) -> float:
    """GB a running job books besides what it uses now (the caller takes the max with the live measurement): the
    declaration until its own peak can be believed (history, or settle_s of running), then
    min(declared, max(history cap, 1.2 x own peak + 1))."""
    s = j["spec"]
    peak = float(j.get("vram_peak") or 0)
    up = vram_raised(s, peak, seen)
    if up:
        return up
    declared = float(s.get("vram_gb") or 0)
    if not trust or s.get("exclusive"):
        return declared
    own = VRAM_MARGIN * peak + VRAM_PAD_GB
    if "vram_gb" in (known or {}):
        return min(declared, max(known["vram_gb"], own))
    if peak <= 0 or now - j.get("t0", 0) < settle_s:
        return declared
    return min(declared, own)


# ------------------------------------------------------------------------------------------------ CPU, RAM
def cpu_charge(j: dict, now: float, known: float = None, young_s: float = 300.0) -> float:
    """Cores a running job is charged: declared while young or unmeasured (at most the history cap), then 1.2 x its
    measured peak over the last few minutes."""
    declared = float(max(int(j["spec"].get("cpu") or 0), 1))
    peak = j.get("cores_peak")
    if peak is None or now - j.get("t0", 0) < young_s:
        return declared if known is None else max(1.0, min(declared, max(known, CPU_MARGIN * float(peak or 0))))
    return max(1.0, CPU_MARGIN * float(peak))


def ram_declared(spec: dict, default: float) -> float:
    return float(spec.get("ram_gb") or 0) or float(default)


def ram_over(spec: dict, rss: float, default: float) -> bool:
    return rss > RAM_OVER_MARGIN * ram_declared(spec, default) + RAM_OVER_PAD_GB


def ram_raised(spec: dict, seen: float, default: float) -> float:
    """seen + 1 when the key's recorded RAM peak broke this declaration, else 0."""
    return round(seen + 1.0, 1) if ram_over(spec, seen, default) else 0.0


def ram_need(spec: dict, known: dict, seen: float, default: float) -> float:
    """GB of RAM a queued job is booked at: declared (else the default), at most history's cap, at least `ram_raised`."""
    need = float(spec.get("ram_gb") or 0) or known.get("ram_gb", default)
    if spec.get("ram_gb") and "ram_gb" in known:
        need = min(need, known["ram_gb"])
    return max(need, ram_raised(spec, seen, default))


def ram_reserve(j: dict, now: float, need: float, young_s: float = 300.0) -> float:
    """RAM still to come from a young job: its booking minus what its tree already holds, tapering to 0 at young_s."""
    age = now - j.get("t0", 0)
    if age >= young_s:
        return 0.0
    return max(0.0, need - float(j.get("rss_now") or 0)) * (1.0 - max(age, 0.0) / young_s)
