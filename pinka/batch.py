"""Foreground mode: `pinka run jobs.json` runs a list of jobs here and returns when they have all ended.

The list is JSON: [{"name": "a", "cmd": [...] or "shell string", "vram_gb": 6, "cpu": 2, "ram_gb": 4,
"after": ["other name"], "tries": 2, "timeout_h": 1, ...}, ...]; any submit field works, and `gpu_gb` / `cpus` are
accepted for vram_gb / cpu. Job ids are the names. The same dispatcher runs them, in a state dir of the list's own
(default: <list>.pinka/ next to it), so packing, booking, retries and markers are the daemon's.

Resume: a rerun of the list skips jobs already done with the same command, gives a job that failed one more try
(--retry-failed: all its tries) and first stops whatever a killed runner left running. SIGTERM / Ctrl-C stops every
job and exits.
"""
from __future__ import annotations

import graphlib
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from . import host
from .config import atomic_json
from .dispatcher import Dispatcher
from .jobs import Spec

ALIASES = {"gpu_gb": "vram_gb", "cpus": "cpu"}
FIELDS = {f.name for f in fields(Spec)}


def load_list(path) -> list:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("a job list is a JSON list of objects")
    out, names = [], set()
    for i, r in enumerate(rows):
        r = {ALIASES.get(k, k): v for k, v in r.items()}
        label = repr(r.get("name", "entry %d" % i))
        bad = sorted(set(r) - FIELDS)
        if bad:
            raise ValueError("job %s: unknown field %s" % (label, ", ".join(bad)))
        if "name" not in r or "cmd" not in r:
            raise ValueError("job %s: needs name and cmd" % label)
        if r["name"] in names or "/" in r["name"]:
            raise ValueError("job %s: name repeated or contains '/'" % label)
        names.add(r["name"])
        out.append(Spec(**r).check())
    for s in out:
        missing = sorted(set(s.after) - names)
        if missing:
            raise ValueError("job %r: unknown after %s" % (s.name, missing))
    try:
        graphlib.TopologicalSorter({s.name: s.after for s in out}).prepare()
    except graphlib.CycleError as e:
        raise ValueError("dependency cycle: %s" % " -> ".join(e.args[1]))
    return out


def run_batch(path, state=None, cfg: dict = None, retry_failed: bool = False, gpu=None, out=sys.stdout) -> int:
    specs = load_list(path)
    state = Path(state or Path(path).with_suffix(".pinka"))
    d = Dispatcher(state, cfg=cfg, gpu=gpu, stop_on_exit=True)
    names = [s.name for s in specs]

    def prepare(d: Dispatcher) -> None:
        jobs, now = d.st["jobs"], time.time()
        for j in d.jobs("running"):                   # left by a runner that was killed: stop, then run again
            try:
                host.stop(d.jdir(j["id"]) / "owned.json", grace=5.0)
            except (RuntimeError, OSError, ValueError) as e:
                raise SystemExit("job %s of an earlier runner survives: %s" % (j["id"], e))
            j["state"] = "cancelled"
        for j in d.jobs("queued"):
            if j["id"] not in names:
                d.finish(j, "cancelled", None, "not in the list")
        for i, s in enumerate(specs):
            old = jobs.get(s.name)
            if old and old["state"] == "done" and old["spec"]["cmd"] == s.cmd:
                continue
            spec = asdict(s)
            if old and old["state"] == "failed" and not retry_failed:
                spec["tries"] = 1
            jobs[s.name] = dict(id=s.name, t_submit=now + i * 1e-6, spec=spec, state="queued", tries=0, why="new")
            atomic_json(d.jdir(s.name) / "spec.json", jobs[s.name])
        d.save()

    rc = d.run(until_done=True, prepare=prepare)
    jobs = [d.st["jobs"][n] for n in names if n in d.st["jobs"]]
    w = max([len(n) for n in names] + [3])
    print("\n%-*s  %-9s %4s %5s %7s  %s" % (w, "job", "state", "card", "tries", "hours", "why"), file=out)
    for j in jobs:
        h = (j.get("t1", time.time()) - j["t0"]) / 3600 if j.get("t0") else 0.0
        print("%-*s  %-9s %4s %5d %7.3f  %s" % (w, j["id"], j["state"], "-" if j.get("gpu") is None else j["gpu"],
                                              j.get("tries", 0), h, j.get("why") or ""), file=out)
    ok = sum(j["state"] == "done" for j in jobs)
    print("%d jobs, %d done; logs and markers under %s/jobs/<name>/" % (len(jobs), ok, state), file=out)
    return rc or (0 if ok == len(names) else 1)
