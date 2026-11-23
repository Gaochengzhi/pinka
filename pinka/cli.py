"""pinka: pack GPU jobs onto shared cards by measured memory.

  pinka daemon [--once]                      the dispatcher (keep it running, e.g. in tmux or a systemd user unit)
  pinka submit --vram GB [flags] -- CMD ...  queue a job; prints its id
  pinka fanout --arms a,b,c [--collect CMD] [flags] -- CMD with {arm}
  pinka run JOBS.json [--gpus 0,1]           run a list of jobs here, in the foreground, and exit when all have ended
  pinka list [--all] | show ID | cancel ID.. | retarget ID.. [--vram GB] [--ram GB] [--gpus 0,1] | wait ID..
  pinka status | usage [--hours H] | history [--all] | config

The state dir is --state-dir, else $PINKA_HOME, else $XDG_STATE_HOME/pinka, else ~/.pinka.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import booking as B
from . import jobs as J
from .config import FINAL, load_config, read_json, state_dir

STALE_S = 120.0


def _ids(s: str) -> list:
    return [x for x in (s or "").split(",") if x]


def _warn(age: float) -> None:
    if age > STALE_S:
        print("note: no dispatcher is running for this state dir (start one: pinka daemon)", file=sys.stderr)


def _hist(a) -> dict:
    return read_json(state_dir(a.state_dir) / "history.json", {}) or {}


def cmd_submit(a, arms=None):
    cmd = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
    if not cmd:
        sys.exit("no command (put it after --)")
    cmd = cmd[0] if len(cmd) == 1 else cmd
    name = a.name or Path(cmd.split()[0] if isinstance(cmd, str) else cmd[0]).name
    defaults = B.history_defaults(_hist(a), name)
    for flag, key in (("vram", "vram_gb"), ("cpu", "cpu"), ("ram", "ram_gb")):
        if getattr(a, flag) is None and key in defaults:
            setattr(a, flag, defaults[key])
            print("%s %s from the history of '%s' (p95 x 1.2)" % (flag, defaults[key], B.hist_prefix(name)), file=sys.stderr)
    if a.vram is None and not a.exclusive:
        sys.exit("declare --vram GB (the job's peak VRAM; 0 for a CPU job) or --exclusive")
    _, status, _, age = J.snapshot(a.state_dir)
    sizes = [c["total_gb"] - (status.get("cfg") or {}).get("headroom_gb", 0) for c in (status.get("cards") or {}).values()]
    if sizes and (a.vram or 0) > max(sizes) and not a.exclusive:
        print("warning: --vram %g is more than any card offers (%.1f GB after headroom): it will never start; use "
              "--exclusive for a whole card" % (a.vram, max(sizes)), file=sys.stderr)
    env = dict(os.environ) if a.copy_env else {}
    env.update(dict(kv.split("=", 1) for kv in a.env))
    kw = dict(key=a.key, owner=a.owner or os.environ.get("USER", ""), cwd=a.cwd or os.getcwd(), log_dir=a.log_dir,
              env=env, vram_gb=a.vram or 0.0, cpu=int(a.cpu or 0), ram_gb=a.ram or 0.0, exclusive=a.exclusive,
              gpus=[int(g) for g in _ids(a.gpus)], tag=a.tag, priority=a.priority, after=_ids(a.after),
              when_exists=a.when_exists, tries=a.tries, timeout_h=a.timeout_h)
    if arms is None:
        print(J.submit(cmd, name=name, state=a.state_dir, **kw))
    else:
        out = J.fanout(cmd, arms, name, collect=a.collect or None, state=a.state_dir, **kw)
        for arm, jid in out["arms"].items():
            print(jid, "%s-%s" % (name, arm))
        if out["collect"]:
            print(out["collect"], "%s-collect" % name)
    _warn(age)
    return 0


def cmd_fanout(a):
    return cmd_submit(a, _ids(a.arms) or sys.exit("--arms a,b,c"))


def _age(t):
    s = time.time() - t
    return "%dm" % (s / 60) if s < 5400 else "%.1fh" % (s / 3600)


def cmd_list(a):
    st, _, inbox, age = J.snapshot(a.state_dir)
    jobs = list(st["jobs"].values()) + [dict(j, state="inbox", why="not yet read by the dispatcher") for j in inbox]
    if not a.all:
        cut = time.time() - 6 * 3600
        jobs = [j for j in jobs if j["state"] not in FINAL or j.get("t1", 0) > cut]
    order = {"running": 0, "queued": 1, "inbox": 2, "failed": 3, "cancelled": 4, "done": 5}
    jobs.sort(key=lambda j: (order.get(j["state"], 9), -float(j["spec"].get("priority") or 0), j["t_submit"]))
    print("%-16s %-9s %4s %-22s %11s %7s %5s  %s" % ("id", "state", "card", "name", "vram b/d", "cpu", "age", "why"))
    for j in jobs:
        s = j["spec"]
        vram = "card" if s.get("exclusive") else "%s/%g" % ("%.1f" % j["vram_booked"] if j.get("vram_booked") is not None
                                                            and j["state"] == "running" else "-", s.get("vram_gb") or 0)
        cpu = "%.1f/%s" % (j["cores_peak"], s.get("cpu") or "-") if j["state"] == "running" and j.get("cores_peak") is not None else str(s.get("cpu") or "")
        print("%-16s %-9s %4s %-22s %11s %7s %5s  %s" % (j["id"][:16], j["state"], "" if j.get("gpu") is None else j["gpu"],
                                                       s["name"][:22], vram, cpu, _age(j.get("t0") or j["t_submit"]),
                                                       (j.get("why") or "")[:120]))
    _warn(age)
    return 0


def cmd_show(a):
    j = J.find(a.id, a.state_dir)
    if j is None:
        sys.exit("no job %s" % a.id)
    print(json.dumps(j, indent=1, default=str))
    log = Path(j["spec"].get("log_dir") or state_dir(a.state_dir) / "jobs" / a.id) / "log.txt"
    if log.exists():
        print("\n--- tail of %s" % log)
        print(log.read_bytes()[-3000:].decode(errors="replace"))
    return 0


def cmd_cancel(a):
    for jid in a.ids:
        J.cancel(jid, a.state_dir)
    print("cancel requested for %d job(s); the dispatcher acts within one round" % len(a.ids))
    return 0


def cmd_retarget(a):
    gpus = None if a.gpus is None else [int(g) for g in _ids(a.gpus)]
    if gpus is None and a.vram is None and a.ram is None:
        sys.exit("pass --gpus, --vram and / or --ram")
    for jid in a.ids:
        J.retarget(jid, a.state_dir, gpus=gpus, vram_gb=a.vram, ram_gb=a.ram)
    return 0


def cmd_wait(a):
    s = J.wait(a.ids, a.poll, a.state_dir)
    for i, v in s.items():
        print(i, v)
    return 0 if all(v == "done" for v in s.values()) else 1


def cmd_status(a):
    st, status, inbox, age = J.snapshot(a.state_dir)
    if not status:
        print("no status yet: no dispatcher has run on %s" % state_dir(a.state_dir))
        return 1
    run = [j for j in st["jobs"].values() if j["state"] == "running"]
    c = status.get("counts", {})
    m = status.get("mem") or {}
    print("dispatcher pid %s, last round %.0f s ago; %d running, %d queued, %d in inbox" % (
        status.get("pid"), age, c.get("running", 0), c.get("queued", 0), len(inbox)))
    print("cpu charged %s / budget %s cores; RAM used %.1f GB, line %.1f GB (%s)" % (
        (status.get("cpu") or {}).get("charged"), (status.get("cpu") or {}).get("budget"), m.get("used", 0), m.get("line", 0),
        m.get("source", "none")))
    hd = (status.get("cfg") or {}).get("headroom_gb", 0)
    print("\n%4s %6s %15s %8s %9s %8s  %s" % ("card", "util%", "used/total GB", "booked", "foreign", "free", "jobs (now/booked GB)"))
    for g, x in sorted(status.get("cards", {}).items(), key=lambda kv: int(kv[0])):
        free = x["total_gb"] - hd - x["foreign_gb"] - x["pool_gb"]
        js = ", ".join("%s %s (%.1f/%.1f)" % (j["id"], j["spec"]["name"][:16], j.get("vram_now") or 0, j.get("vram_booked") or 0)
                       for j in run if str(j.get("gpu")) == str(g))
        print("%4s %6d %15s %8.1f %9.1f %8.1f  %s" % (g, x["util"], "%.1f/%.1f" % (x["used_gb"], x["total_gb"]), x["pool_gb"],
                                                   x["foreign_gb"], max(free, 0.0), js or "-"))
    cpu_jobs = [j for j in run if j.get("gpu") is None]
    if cpu_jobs:
        print("CPU jobs: " + ", ".join("%s %s" % (j["id"], j["spec"]["name"]) for j in cpu_jobs))
    idle = [g for g in (status.get("idle") or {}) if not any(str(j.get("gpu")) == g for j in run)]
    ready = [j for j in st["jobs"].values() if j["state"] == "queued" and B.is_gpu(j["spec"]) and j.get("cause") not in
             ("after", "gate", "new")]
    if idle and ready:
        print("\nidle card(s) %s while %d GPU job(s) wait; `pinka list` shows why" % (",".join(idle), len(ready)))
    _warn(age)
    return 0


def cmd_usage(a):
    now, rows = time.time(), []
    try:
        for ln in (state_dir(a.state_dir) / "usage.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if r.get("t", 0) >= now - a.hours * 3600:
                rows.append(r)
    except (OSError, ValueError):
        pass
    if not rows:
        print("no usage samples in the last %g h" % a.hours)
        return 0
    split, card_h = {}, 0.0
    for r, nxt in zip(rows, rows[1:] + [None]):
        dt = min((nxt["t"] - r["t"]) if nxt else 60, 300) / 3600
        for util, used, booked, jobs in r["cards"].values():
            card_h += dt
            k = ("computing" if util >= 10 else "job, GPU idle") if jobs else (
                "queued: " + max(r["q_gpu"], key=r["q_gpu"].get) if r.get("q_gpu") else "no work queued")
            split[k] = split.get(k, 0.0) + dt
    print("last %g h: %.2f card-hours\n\n%-28s %10s %6s" % (a.hours, card_h, "card state", "card-hours", "share"))
    for k, v in sorted(split.items(), key=lambda kv: -kv[1]):
        print("%-28s %10.2f %5.0f%%" % (k, v, 100 * v / max(card_h, 1e-9)))
    return 0


def cmd_history(a):
    hist = _hist(a)
    print("%-28s %5s %18s %12s %12s" % ("key", "runs", "VRAM p95/max GB", "cores max", "RAM max GB"))
    for k, e in sorted(hist.items()):
        v = e.get("vram") or []
        if not a.all and len(v) < 1:
            continue
        print("%-28s %5d %18s %12s %12s" % (k[:28], len(v), "%.1f/%.1f" % (B._p95(v), max(v)) if v else "-",
                                         "%.1f" % max(e["cores"]) if e.get("cores") else "-",
                                         "%.1f" % max(e["ram"]) if e.get("ram") else "-"))
    return 0


def cmd_config(a):
    print(json.dumps(load_config(state_dir(a.state_dir)), indent=1))
    return 0


def cmd_daemon(a):
    from .dispatcher import Dispatcher
    return Dispatcher(a.state_dir).run(once=a.once)


def cmd_run(a):
    from .batch import run_batch
    cfg = {k: v for k, v in (("poll_s", a.poll), ("cpu_budget", a.cpus), ("ram_budget_gb", a.ram_gb)) if v}
    cfg["idle_s"] = 0.0                                # the runner owns its cards: an empty card is idle at once
    if a.gpus is not None:
        cfg["cards"] = [int(g) for g in _ids(a.gpus)]
    try:
        return run_batch(a.jobs, a.state_dir, cfg=cfg, retry_failed=a.retry_failed)
    except ValueError as e:
        sys.exit(str(e))


def _submit_flags(s):
    s.add_argument("--name", default="", help="job name (default: the command's first word); also its history key")
    s.add_argument("--key", default="", help="history key, when names differ but resource use does not")
    s.add_argument("--vram", type=float, default=None, help="peak VRAM GB on its card; 0 = CPU job (default: history)")
    s.add_argument("--cpu", type=int, default=None, help="cores; > 0 pins the job to that many free cores (Linux)")
    s.add_argument("--ram", type=float, default=None, help="host RAM GB")
    s.add_argument("--exclusive", action="store_true", help="alone on its card")
    s.add_argument("--gpus", default="", help="allowed cards, e.g. 0,2 (default: any)")
    s.add_argument("--tag", default="", help="per-card cap by tag (config tag_caps)")
    s.add_argument("--priority", type=float, default=0.0, help="higher starts first")
    s.add_argument("--after", default="", help="ids that must be done first (comma-separated)")
    s.add_argument("--when-exists", default="", help="stay queued until this path exists")
    s.add_argument("--tries", type=int, default=1)
    s.add_argument("--timeout-h", type=float, default=0.0)
    s.add_argument("--env", action="append", default=[], help="K=V (repeatable)")
    s.add_argument("--no-copy-env", dest="copy_env", action="store_false",
                   help="run with the dispatcher's environment, not this shell's")
    s.add_argument("--cwd", default="")
    s.add_argument("--log-dir", default="", help="log.txt, STATUS, DONE / ERROR here (default <state>/jobs/<id>)")
    s.add_argument("--owner", default="")
    s.add_argument("argv", nargs=argparse.REMAINDER, metavar="CMD")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pinka", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=None, help="default: $PINKA_HOME, $XDG_STATE_HOME/pinka, ~/.pinka")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon").add_argument("--once", action="store_true")
    _submit_flags(sub.add_parser("submit"))
    s = sub.add_parser("fanout")
    s.add_argument("--arms", required=True)
    s.add_argument("--collect", default="", help="shell command run after every arm is done ({arms} = all arms)")
    _submit_flags(s)
    s = sub.add_parser("run")
    s.add_argument("jobs")
    s.add_argument("--gpus", default=None, help="cards to use (default: CUDA_VISIBLE_DEVICES, else all)")
    s.add_argument("--cpus", type=float, default=None, help="CPU budget (default: the cores this process may use)")
    s.add_argument("--ram-gb", type=float, default=None, help="RAM budget (default: measure the host)")
    s.add_argument("--retry-failed", action="store_true")
    s.add_argument("--poll", type=float, default=2.0)
    sub.add_parser("list").add_argument("--all", action="store_true")
    sub.add_parser("show").add_argument("id")
    sub.add_parser("cancel").add_argument("ids", nargs="+")
    s = sub.add_parser("retarget")
    s.add_argument("ids", nargs="+")
    s.add_argument("--gpus", default=None, help="allowed cards ('' = any)")
    s.add_argument("--vram", type=float, default=None)
    s.add_argument("--ram", type=float, default=None)
    s = sub.add_parser("wait")
    s.add_argument("ids", nargs="+")
    s.add_argument("--poll", type=float, default=30.0)
    sub.add_parser("status")
    sub.add_parser("usage").add_argument("--hours", type=float, default=24.0)
    sub.add_parser("history").add_argument("--all", action="store_true")
    sub.add_parser("config")
    a = ap.parse_args(argv)
    return globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
