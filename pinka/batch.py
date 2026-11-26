"""Job lists. `pinka run jobs.json` runs a list here, in the foreground, and returns when its jobs have all ended;
`pinka submit --list jobs.json` (submit_list) puts the same list into the shared dispatcher instead.

The list is JSON: [{"name": "a", "cmd": [...] or "shell string", "vram_gb": 6, "cpu": 2, "ram_gb": 4,
"after": ["other name"], "tries": 2, "timeout_h": 1, ...}, ...]; any submit field works, and `gpu_gb` / `cpus` are
accepted for vram_gb / cpu. Job ids are the names. The same dispatcher runs them, in a state dir of the list's own
(default: <list>.pinka/ next to it), so packing, booking, retries and markers are the daemon's.

Resume: a rerun of the list skips jobs already done with the same command, gives a job that failed one more try
(--retry-failed: all its tries) and first stops whatever a killed runner left running. SIGTERM / Ctrl-C stops every
job and exits.

submit_list: every job gets a fresh pool id, `after` names are mapped to those ids (an `after` that is no name in the
list must be the id of a job the pool knows), names can be prefixed, and the name -> id map is written to a JSON file
(default: <list>.ids.json). Resubmitting with the same map keeps the jobs that are queued, running or done with the
same command and dependencies and submits the rest anew.
"""
from __future__ import annotations

import graphlib
import json
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from . import host
from .config import atomic_json, read_json, state_dir
from .dispatcher import Dispatcher
from . import jobs as J
from .jobs import Spec

ALIASES = {"gpu_gb": "vram_gb", "cpus": "cpu"}
FIELDS = {f.name for f in fields(Spec)}


def load_list(path, known=None) -> list:
    """Read and check a job list file; returns its Specs. known(ids) -> set of the ids that may appear in `after`
    without being a name in the list (default: none)."""
    return parse_list(json.loads(Path(path).read_text()), known)


def parse_list(rows, known=None) -> list:
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
        if not isinstance(r["name"], str) or not r["name"] or r["name"] in names or "/" in r["name"]:
            raise ValueError("job %s: name repeated or contains '/'" % label)
        names.add(r["name"])
        out.append(Spec(**r).check())
    outside = sorted({a for s in out for a in s.after} - names)
    ok = set(known(outside)) if known and outside else set()
    for s in out:
        missing = sorted(set(s.after) - names - ok)
        if missing:
            raise ValueError("job %r: unknown after %s" % (s.name, missing))
    try:
        graphlib.TopologicalSorter({s.name: [a for a in s.after if a in names] for s in out}).prepare()
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


KEEP = ("inbox", "queued", "running", "done")                 # states a resubmitted job is kept in


def ids_path(path) -> Path:
    """Default name -> id map of a list file: jobs.json -> jobs.ids.json, next to it."""
    return Path(path).with_suffix(".ids.json")


def submit_list(jobs, state=None, prefix: str = "", ids_file=None, cwd: str = "", env: dict = None,
                copy_env: bool = True, owner: str = "", log_dir: str = "", priority: float = None, tag: str = "",
                after=(), dry: bool = False, out=None) -> dict:
    """Put a job list into the shared dispatcher of `state`; returns {name: id} (names without the prefix).

    jobs: a list file (path) or the list itself (dicts in the `pinka run` format). Each job's name becomes
    prefix + name (its history key stays the unprefixed name unless it sets `key`). Defaults for fields a job leaves
    out: cwd (else the current directory), owner (else $USER), log_dir (a job logs to <log_dir>/<name>), priority,
    tag; env is os.environ (copy_env) + env + the job's env. after: pool ids that every job without a dependency in
    the list waits for (the list's roots). ids_file (default <list>.ids.json for a path, none for a
    list) gets {"state_dir", "prefix", "ids"} (ids in dependency order); when it already holds a map for this state dir and prefix, a job
    whose recorded id is queued, running or done with the same command and dependencies is kept, a queued one that
    changed is cancelled and submitted anew, and a running one that changed is an error (cancel it first).
    dry: plan only; returns the map the submission would give, unsubmitted jobs as "new:<name>"."""
    sd = state_dir(state)
    rows = json.loads(Path(jobs).read_text()) if isinstance(jobs, (str, os.PathLike)) else jobs
    if ids_file is None and isinstance(jobs, (str, os.PathLike)):
        ids_file = ids_path(jobs)
    known = lambda xs: [x for x, st in J.job_states(xs, sd).items() if st != "unknown"]  # noqa: E731
    specs = {s.name: s for s in parse_list(rows, known)}
    after = [a for a in after if a]
    unknown = sorted(set(after) - set(known(after))) if after else []
    if unknown:
        raise ValueError("after: no such job in the pool: %s" % ", ".join(unknown))
    for s in specs.values():
        if after and not any(a in specs for a in s.after):
            s.after = list(s.after) + [a for a in after if a not in s.after]
    order = list(graphlib.TopologicalSorter({n: [a for a in s.after if a in specs] for n, s in specs.items()}).static_order())
    old = read_json(ids_file, None) if ids_file else None
    old = old["ids"] if old and old.get("state_dir") == str(sd.resolve()) and old.get("prefix") == prefix else {}
    base_env = dict(os.environ) if copy_env else {}
    base_env.update(env or {})
    ids, plan, cancel, stuck = {}, [], [], []
    for n in order:
        s = specs[n]
        spec = asdict(s)
        spec.update(name=prefix + n, key=s.key or (n if prefix else ""), cwd=s.cwd or cwd or os.getcwd(),
                    owner=s.owner or owner or os.environ.get("USER", ""), env=dict(base_env, **s.env),
                    after=[ids.get(a, a) for a in s.after], tag=s.tag or tag,
                    log_dir=s.log_dir or (str(Path(log_dir) / n) if log_dir else ""))
        if priority is not None and "priority" not in next(r for r in rows if r["name"] == n):
            spec["priority"] = priority
        rec = J.find(old[n], sd) if n in old else None
        if rec and rec.get("state") in KEEP and rec["spec"]["cmd"] == spec["cmd"] and rec["spec"]["after"] == spec["after"]:
            ids[n] = old[n]
            plan.append((n, "kept", rec["state"]))
            continue
        if rec and rec.get("state") == "running":
            stuck.append("%s (%s)" % (n, old[n]))
        elif rec and rec.get("state") in ("inbox", "queued"):
            cancel.append(old[n])
        ids[n] = "new:" + n
        plan.append((n, "new", rec.get("state") if rec else ""))
        spec["_new"] = True
        specs[n] = spec
    if stuck:
        raise ValueError("running with a different command or dependencies: %s; cancel them first (pinka cancel ID)"
                         % ", ".join(stuck))
    if not dry:
        for jid in cancel:
            J.cancel(jid, sd)
        for n in order:                                         # dependencies first, so `after` holds real ids
            spec = specs[n]
            if isinstance(spec, dict):
                spec.pop("_new")
                spec["after"] = [ids.get(a[4:], a) if a.startswith("new:") else a for a in spec["after"]]
                cmd, name = spec.pop("cmd"), spec.pop("name")
                ids[n] = J.submit(cmd, name=name, state=sd, **spec)
        if ids_file:
            atomic_json(ids_file, dict(state_dir=str(sd.resolve()), prefix=prefix, ids={n: ids[n] for n in order}))
    if out is not None:
        for n, what, was in plan:
            print("%s %s %s%s" % (ids[n], prefix + n, what, " (was %s)" % was if was else ""), file=out)
    return {n: ids[n] for n in order}
