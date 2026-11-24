"""State directory, configuration and small file helpers.

The state directory is $PINKA_HOME, else $XDG_STATE_HOME/pinka, else ~/.pinka. Its config.json (optional) overrides
DEFAULTS and is re-read by the dispatcher every round, so a change takes effect without a restart.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

DEFAULTS = dict(
    poll_s=10.0,            # seconds between dispatcher rounds
    fast_poll_s=3.0,        # ... while host memory is close to the line
    gpu_backend="auto",     # "auto" (pynvml if installed, else nvidia-smi), "nvml", "nvidia-smi", "none", or
                            # "fake:2x24" (two fake 24 GB cards, for tests and dry runs); env PINKA_GPU_BACKEND wins
    smi_age_s=10.0,         # nvidia-smi answers are reused this long
    cards=None,             # card indices the pool may use (None: all, or the CUDA_VISIBLE_DEVICES the dispatcher starts with)
    headroom_gb=2.0,        # VRAM kept free on every card
    max_starts=4,           # jobs started per round (GPU and CPU jobs counted apart)
    hold_s=900.0,           # a job blocked this long at the head of the queue reserves the card closest to fitting it
    idle_s=120.0,           # a card with no job for this long is idle (see admit_idle in AGENTS.md)
    idle_vram_gb=1.0,       # ... and uses at most this much VRAM outside the pool
    trust_measured=True,    # book by measured peaks (False: always book the declaration)
    settle_s=600.0,         # a job without history books its declaration this long, then its own measured peak
    young_s=300.0,          # CPU is charged as declared, and RAM still to come is reserved, while a job is this young
    hist_min_n=3,           # finished runs of a history key before their peaks replace a declaration
    hist_min_wall_s=60.0,   # shorter successful runs do not enter history
    cpu_budget=0.0,         # cores the pool may charge (0: cores this host / cgroup gives x cpu_overcommit)
    cpu_overcommit=1.0,
    pin_cpus=True,          # a job with cpu > 0 is pinned to that many free cores (Linux)
    memory_source="auto",   # "auto" (cgroup v2 limit, else host RAM), "cgroup", "host", "none"
    ram_budget_gb=0.0,      # > 0: book RAM against this budget (declared or measured, the larger) instead of measuring the host
    ram_frac=0.9,           # admission line: used + young + need <= ram_frac x memory limit
    ram_default_gb=1.0,     # RAM an undeclared job counts as
    ram_stop_gb=None,       # below this margin under the line, stop the job furthest above its RAM declaration
                            # (None: 4 % of the limit; 0: never)
    ram_retries=1,          # requeues of a job stopped by that rule, with its RAM raised to its measured peak
    oom_retries=1,          # requeues after a CUDA out-of-memory error or a VRAM stop, beyond `tries`
    kill_retries=2,         # requeues after the job's root dies of SIGKILL without a pool stop (host OOM killer)
    over_stop_free_gb=1.0,  # a job above its VRAM declaration on a card with less free than this is stopped (0: never)
    tag_caps={},            # per-card caps by job tag, e.g. {"train": 2}
    conc=True,              # book same-key jobs on a card together, by their measured concurrent use (AGENTS.md)
    conc_margin=1.1,
    conc_hours=72.0,        # ... learned from this much of usage.jsonl
    conc_min=30,            # ... card-minutes at one job count before that count is trusted
    conc_q=0.99,
    conc_every_s=1800.0,    # ... re-learned this often
    low_memory_gb=0.0,      # call the low_memory hook while the margin under the RAM line is below this (0: never)
    low_memory_every_s=30.0,
    keep_days=14.0,         # finished jobs leave state.json after this many days (their jobs/<id>/ folder stays)
    hooks={},               # {"job_env": "module:function", "try_end": ..., "low_memory": ...}; see AGENTS.md, Hooks
)
FINAL = ("done", "failed", "cancelled")


def state_dir(path=None) -> Path:
    if path:
        return Path(path).expanduser()
    if os.environ.get("PINKA_HOME"):
        return Path(os.environ["PINKA_HOME"]).expanduser()
    if os.environ.get("XDG_STATE_HOME"):
        return Path(os.environ["XDG_STATE_HOME"]) / "pinka"
    return Path.home() / ".pinka"


def load_config(state: Path, over: dict = None) -> dict:
    c = dict(DEFAULTS)
    c.update(read_json(Path(state) / "config.json", {}) or {})
    if os.environ.get("PINKA_GPU_BACKEND"):
        c["gpu_backend"] = os.environ["PINKA_GPU_BACKEND"]
    c.update(over or {})
    return c


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic_json(path, value) -> None:
    """Write JSON so that a reader sees the old file or the new one, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, str(path))
