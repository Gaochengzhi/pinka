"""The dispatcher: the only writer of state.json. One per state directory (dispatch.lock).

Every round: read submissions, cancels and retargets; reap jobs whose process tree is gone (retry, requeue or finish
them); measure what running jobs use (VRAM per process, cores, private memory); stop a job that broke its declaration
when its card or the host runs out; start queued jobs, highest priority first, on the card with room that is least
loaded (fewest jobs), best fit breaking ties. README.md describes the rules.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import traceback
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import booking as B
from . import gpu as G
from . import host
from .config import FINAL, atomic_json, load_config, read_json, state_dir
from .hooks import Hooks
from .jobs import Spec

OOM_RE = re.compile(rb"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|CUDA_ERROR_OUT_OF_MEMORY|"
                    rb"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED")
OOM_SCAN_BYTES = 1 << 20
KILL_RCS = (137, -9)
OVER_STEP_GB, RAM_OVER_STEP_GB = 2.0, 8.0      # an over-declaration event repeats when the peak grew this much
CPU_WINDOW_S = 300.0                           # cores are charged at the peak of this window
USAGE_EVERY_S = 60.0
PRUNE_EVERY_S = 3600.0
CAUSES = (("after ", "after"), ("waiting for ", "gate"), ("start limit", "start_limit"), ("CPU ", "cpu"),
          ("memory ", "ram"), ("never fits", "never_fits"), ("VRAM ", "vram"), ("exclusive", "exclusive"),
          ("tag ", "tag"), ("reserved for", "reserved"), ("card not allowed", "pinned"), ("no card", "no_card"),
          ("cores to pin", "cpu_pin"), ("retry", "retry"), ("new", "new"))
NOT_READY = ("after", "gate", "new")
SHELL = shutil.which("bash") or "/bin/sh"


def wait_cause(why: str) -> str:
    """One word for why a queued job waits (the first card it may use decides)."""
    cs = [next((c for k, c in CAUSES if k in part), "other") for part in (why or "").split("; card")]
    return next((c for c in cs if c != "pinned"), cs[0])


@dataclass
class Acct:
    """One card's accounting in a round."""
    index: int
    total_gb: float
    used_gb: float
    util: int = 0
    foreign_gb: float = 0.0          # used by processes outside the pool
    pool_gb: float = 0.0             # booked by pool jobs: sum of max(measured now, booking)
    jobs: int = 0
    foreign_procs: int = 0
    exclusive: bool = False
    reserved_for: str = ""
    tags: dict = field(default_factory=dict)

    def avail(self, headroom: float) -> float:
        return self.total_gb - headroom - self.foreign_gb - self.pool_gb


def fits(s: Spec, a: Acct, cfg: dict, need: float) -> str:
    """'' when the job fits card `a`, else why not."""
    if s.gpus and a.index not in s.gpus:
        return "card not allowed"
    if a.exclusive:
        return "exclusive job on card"
    if s.exclusive and (a.jobs or a.foreign_gb > cfg["idle_vram_gb"]):
        return "exclusive: card not empty"
    need = a.total_gb - cfg["headroom_gb"] if s.exclusive else need
    av = a.avail(cfg["headroom_gb"])
    if av < need:
        return "VRAM %.1f GB free < %.1f" % (max(av, 0.0), need)
    cap = (cfg.get("tag_caps") or {}).get(s.tag) if s.tag else None
    if cap is not None and a.tags.get(s.tag, 0) >= cap:
        return "tag %s: %d on card, cap %d" % (s.tag, a.tags.get(s.tag, 0), cap)
    return ""


def choose(s: Spec, cards: list, cfg: dict, jid: str, need: float):
    """(best card or None, {card: why not})."""
    why, ok = {}, []
    for a in cards:
        if a.reserved_for and a.reserved_for != jid:
            why[a.index] = "reserved for blocked job " + a.reserved_for
            continue
        w = fits(s, a, cfg, need)
        if w:
            why[a.index] = w
        else:
            ok.append(a)
    if not ok:
        return None, why
    return min(ok, key=lambda a: (a.jobs + a.foreign_procs, a.avail(cfg["headroom_gb"]) - need)), why


class Dispatcher:
    def __init__(self, state=None, cfg: dict = None, gpu=None, memory_fn=None, stop_on_exit: bool = False):
        self.dir = state_dir(state)
        self.st = read_json(self.dir / "state.json", None) or {"jobs": {}}
        self.cfg_over, self.gpu, self.memory_fn, self.stop_on_exit = cfg or {}, gpu, memory_fn, stop_on_exit
        self.c = load_config(self.dir, self.cfg_over)
        self.hooks = Hooks(self.log)
        self.children, self.cpu_prev, self.cpu_win, self.idle_since = {}, {}, {}, {}
        self.hist, self.last, self.cpu_info, self.mem = {}, {}, {}, dict(limit=0.0, used=0.0, line=0.0, source="none")
        self.margin = float("inf")
        self.usage_t = self.prune_t = self.lowmem_t = 0.0
        self.mask = None
        self.stopping, self.started = False, 0
        for sub in ("inbox", "cancel", "retarget", "jobs"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------ helpers
    def log(self, kind: str, **kw) -> None:
        line = json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), kind=kind, **kw), default=str)
        with (self.dir / "events.jsonl").open("a") as f:
            f.write(line + "\n")

    def jobs(self, *states) -> list:
        return [j for j in self.st["jobs"].values() if j["state"] in states]

    def jdir(self, jid: str) -> Path:
        return self.dir / "jobs" / jid

    def logdir(self, j: dict) -> Path:
        return Path(j["spec"].get("log_dir") or self.jdir(j["id"]))

    def status_line(self, j: dict, text: str) -> None:
        d = self.logdir(j)
        d.mkdir(parents=True, exist_ok=True)
        (d / "STATUS").write_text("%s %s (%s): %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), j["spec"]["name"], j["id"], text))

    def state_of(self, jid: str) -> str:
        j = self.st["jobs"].get(jid)
        if j:
            return j["state"]
        return (read_json(self.jdir(jid) / "job.json") or {}).get("state", "unknown id")

    # ------------------------------------------------------------------------------------------ inbox, cancel, retarget
    def ingest(self) -> None:
        for p in sorted((self.dir / "inbox").glob("*.json")):
            j = read_json(p)
            p.unlink()
            if not j or j.get("id") in self.st["jobs"]:
                continue
            j.update(state="queued", tries=0, why="new")
            self.st["jobs"][j["id"]] = j
            atomic_json(self.jdir(j["id"]) / "spec.json", j)
            self.status_line(j, "queued")
            self.log("submit", id=j["id"], name=j["spec"]["name"], owner=j["spec"].get("owner"))
        for p in sorted((self.dir / "retarget").glob("*")):
            d, j = read_json(p, {}) or {}, self.st["jobs"].get(p.name)
            p.unlink()
            if j is not None and j["state"] == "queued":
                for k in ("gpus", "vram_gb", "ram_gb"):
                    if k in d:
                        j["spec"][k] = [int(g) for g in d[k]] if k == "gpus" else float(d[k])
                j.pop("never", None)
                self.log("retarget", id=j["id"], **{k: j["spec"].get(k) for k in d})
        for p in sorted((self.dir / "cancel").glob("*")):
            j = self.st["jobs"].get(p.name)
            p.unlink()
            if j is None or j["state"] in FINAL:
                continue
            if j["state"] == "queued":
                self.finish(j, "cancelled", None, "cancelled while queued")
            else:
                j["stop"] = "cancelled"

    # ------------------------------------------------------------------------------------------ end of tries
    def finish(self, j: dict, state: str, rc, why: str) -> None:
        now = time.time()
        j.update(state=state, rc=rc, t1=now, why=why)
        info = dict(id=j["id"], name=j["spec"]["name"], state=state, rc=rc, tries=j["tries"], gpu=j.get("gpu"),
                    wall_s=round(now - j.get("t0", now), 1), why=why)
        tail = ""
        try:
            tail = (self.logdir(j) / "log.txt").read_bytes()[-3000:].decode(errors="replace")
        except OSError:
            pass
        for d in {self.jdir(j["id"]), self.logdir(j)}:   # markers in the job folder and in its log dir
            if state == "done":
                atomic_json(d / "DONE", info)
            else:
                d.mkdir(parents=True, exist_ok=True)
                (d / "ERROR").write_text("# job %s %s: %s (rc %s, try %d) %s\n\n%s\n" % (
                    j["id"], state, why, rc, j["tries"], time.strftime("%Y-%m-%d %H:%M:%S"), tail))
        if state == "done" and info["wall_s"] >= self.c["hist_min_wall_s"]:
            B.hist_record(self.dir / "history.json", j["spec"], float(j.get("vram_peak") or 0),
                          float(j.get("cores_max") or 0), float(j.get("rss_peak") or 0))
        else:
            self.note_over(j)
        self.status_line(j, state + (": " + why if why else ""))
        atomic_json(self.jdir(j["id"]) / "job.json", j)
        self.log("end", id=j["id"], name=j["spec"]["name"], state=state, rc=rc, why=why, gpu=j.get("gpu"),
                 wall_s=info["wall_s"], vram_peak=j.get("vram_peak"), cores_max=j.get("cores_max"), rss_peak=j.get("rss_peak"))

    def note_over(self, j: dict) -> None:
        """A job that broke its VRAM or RAM declaration leaves its peak in history whatever its end, so the next job of
        its key is booked by what it measured."""
        s, cfg = j["spec"], self.c
        v, r = float(j.get("vram_peak") or 0), float(j.get("rss_peak") or 0)
        v = v if B.over_declared(s, v) and j.get("over_rec") != v else 0.0
        r = r if B.ram_over(s, r, cfg["ram_default_gb"]) and j.get("ram_rec") != r else 0.0
        if v or r:
            B.hist_record(self.dir / "history.json", s, v, 0.0, r)
            j["over_rec"], j["ram_rec"] = v or j.get("over_rec"), r or j.get("ram_rec")

    def in_log(self, j: dict, pattern) -> bool:
        """Does this try's part of log.txt match `pattern`?"""
        try:
            with (self.logdir(j) / "log.txt").open("rb") as f:
                size = f.seek(0, 2)
                f.seek(max(min(int(j.get("log_off") or 0), size), size - OOM_SCAN_BYTES))
                return bool(pattern.search(f.read()))
        except OSError:
            return False

    def requeue(self, j: dict, why: str, counter: str = "") -> None:
        if counter:
            j[counter] = int(j.get(counter) or 0) + 1
        j.update(state="queued", why="retry after " + why)
        self.status_line(j, "queued: retry after " + why)
        self.log("retry", id=j["id"], why=why, tries=j["tries"])

    def ended(self, j: dict, rc) -> None:
        """A try's whole tree is gone: retry, requeue (not counted against tries) or finish."""
        s, cfg, stop = j["spec"], self.c, j.pop("stop", None)
        self.hooks.call(cfg, "try_end", j, rc)
        vstop, rstop = str(stop or "").startswith("VRAM "), str(stop or "").startswith("RAM ")
        oom = vstop or (rc not in (0, None) and not stop and self.in_log(j, OOM_RE))
        killed = rc in KILL_RCS and not stop and not oom
        if oom or killed or rstop:
            self.note_over(j)
            self.log("oom" if oom else "ram_stop_end" if rstop else "kill", id=j["id"], name=s["name"], gpu=j.get("gpu"),
                     rc=rc, vram_peak=j.get("vram_peak"), rss_peak=j.get("rss_peak"))
        if oom and int(j.get("oom_tries") or 0) < int(cfg["oom_retries"]):
            self.requeue(j, stop or "CUDA out of memory (rc %s)" % rc, "oom_tries")
        elif killed and int(j.get("kill_tries") or 0) < int(cfg["kill_retries"]):
            self.requeue(j, "SIGKILL (rc %s), not counted" % rc, "kill_tries")
        elif rstop and int(j.get("ram_tries") or 0) < int(cfg["ram_retries"]):
            s["ram_gb"] = round(max(float(s.get("ram_gb") or 0), float(j.get("rss_peak") or 0) + 1.0), 1)
            self.requeue(j, "%s; RAM declaration raised to %.1f GB" % (stop, s["ram_gb"]), "ram_tries")
        elif stop:
            self.finish(j, "cancelled" if stop == "cancelled" else "failed", rc, stop)
        elif rc == 0:
            self.finish(j, "done", rc, "")
        elif j["tries"] - sum(int(j.get(k) or 0) for k in ("oom_tries", "kill_tries", "ram_tries")) < int(s.get("tries") or 1):
            self.requeue(j, "rc %s" % rc)
        else:
            self.finish(j, "failed", rc, "rc %s%s" % (rc, " (CUDA out of memory)" if oom else " (SIGKILL)" if killed else ""))

    def reap(self, rows: dict) -> tuple:
        """Refresh running trees, stop what must stop, end tries whose tree is empty. Returns ({id: pids}, rows)."""
        live = {}
        for j in self.jobs("running"):
            jid, d = j["id"], self.jdir(j["id"])
            p = self.children.get(jid)
            if p is not None and p.poll() is not None:
                rows = host.processes()                   # the root was a zombie until now
            try:
                rec, rows = host.refresh(d / "owned.json", rows)
            except (OSError, ValueError):
                rec = None
            members = host.live_members(rec, rows) if rec else []
            root_alive = bool(rec) and host.same(rec["root"], rows)
            th = float(j["spec"].get("timeout_h") or 0)
            if root_alive and not j.get("stop") and th and time.time() - j["t0"] > th * 3600:
                j["stop"] = "timeout after %g h" % th
            if members and (j.get("stop") or not root_alive):
                self.log("stop", id=jid, pids=members, why=j.get("stop") or "root exited, members left")
                try:
                    host.stop(d / "owned.json")
                except (RuntimeError, OSError, ValueError) as e:
                    self.log("stop_failed", id=jid, err=str(e))
                    live[jid] = members
                    continue
                rows, members = host.processes(), []
            if members:
                live[jid] = members
                continue
            self.children.pop(jid, None)
            try:
                rc = int((d / ("rc.%d" % j["tries"])).read_text().strip())
            except (OSError, ValueError):
                rc = None
            self.ended(j, rc)
        return live, rows

    # ------------------------------------------------------------------------------------------ measurement
    def measure(self, live: dict, rows: dict) -> None:
        """Cores (utime + stime deltas over the tree, peak over CPU_WINDOW_S) and private memory of each running job."""
        now = time.time()
        for jid in [k for k in self.cpu_prev if k not in live]:
            self.cpu_prev.pop(jid, None)
            self.cpu_win.pop(jid, None)
        for jid, mem in live.items():
            j = self.st["jobs"][jid]
            cur = {p: rows[p]["cpu_s"] for p in mem if p in rows}
            prev, tp = self.cpu_prev.get(jid, (None, 0.0))
            self.cpu_prev[jid] = (cur, now)
            if prev is not None and now - tp >= 1.0:
                cores = sum(max(t - prev.get(p, 0.0), 0.0) for p, t in cur.items()) / (now - tp)
                w = self.cpu_win.setdefault(jid, deque())
                w.append((now, cores))
                while w and now - w[0][0] > CPU_WINDOW_S:
                    w.popleft()
                j["cores_now"], j["cores_peak"] = round(cores, 1), round(max(c for _, c in w), 1)
                j["cores_max"] = max(float(j.get("cores_max") or 0), j["cores_peak"])
            j["rss_now"] = round(host.rss_gb(mem, rows), 2)
            j["rss_peak"] = max(float(j.get("rss_peak") or 0), j["rss_now"])

    def read_memory(self) -> None:
        cfg = self.c
        if cfg.get("ram_budget_gb"):
            used = sum(max(float(j.get("rss_now") or 0), float(j.get("ram_booked") or 0)) for j in self.jobs("running"))
            self.mem = dict(limit=float(cfg["ram_budget_gb"]), used=used, line=float(cfg["ram_budget_gb"]), source="budget")
        else:
            limit, used, src = (self.memory_fn or host.memory)(cfg["memory_source"])
            self.mem = dict(limit=limit, used=used, line=cfg["ram_frac"] * limit if limit else 0.0, source=src)
        self.margin = self.mem["line"] - self.mem["used"] if self.mem["line"] else float("inf")

    def ram_stop_gb(self) -> float:
        v = self.c.get("ram_stop_gb")
        return 0.04 * self.mem["limit"] if v is None else float(v)

    def guard_ram(self) -> None:
        """Jobs above their RAM declaration are reported (event ram_over, STATUS); under ram_stop_gb of margin the one
        furthest above is stopped now, so the kernel's OOM killer does not pick a well-behaved neighbour."""
        cfg, over = self.c, []
        for j in self.jobs("running"):
            s, rss = j["spec"], float(j.get("rss_now") or 0)
            dec = max(B.ram_declared(s, cfg["ram_default_gb"]), float(j.get("ram_booked") or 0))
            if not B.ram_over(dict(s, ram_gb=dec), rss, cfg["ram_default_gb"]):
                continue
            over.append((rss - dec, dec, rss, j))
            if rss >= float(j.get("ram_over") or 0) + RAM_OVER_STEP_GB:
                j["ram_over"] = rss
                self.log("ram_over", id=j["id"], name=s["name"], declared=dec, rss=rss, margin=round(self.margin, 1))
                self.status_line(j, "running: RAM %.1f GB > %.1f GB declared (--ram)" % (rss, dec))
        over = [x for x in over if not x[3].get("stop")]
        stop_gb = self.ram_stop_gb()
        if not over or not stop_gb or self.margin >= stop_gb:
            return
        _, dec, rss, j = max(over, key=lambda x: x[0])
        j["stop"] = "RAM %.1f GB > %.1f GB declared with %.1f GB under the line" % (rss, dec, max(self.margin, 0.0))
        self.log("ram_stop", id=j["id"], name=j["spec"]["name"], why=j["stop"])
        try:
            host.stop(self.jdir(j["id"]) / "owned.json", grace=3.0)
        except (RuntimeError, OSError, ValueError) as e:
            self.log("stop_failed", id=j["id"], err=str(e))

    # ------------------------------------------------------------------------------------------ cards
    def backend(self):
        if self.gpu is None:
            self.gpu = G.backend(self.c["gpu_backend"], self.c["smi_age_s"])
        return self.gpu

    def card_mask(self, cards: list):
        """Cards the pool may use: config `cards`, else the CUDA_VISIBLE_DEVICES the dispatcher started with."""
        if self.c.get("cards") is not None:
            return set(int(x) for x in self.c["cards"])
        if self.mask is None:
            env = os.environ.get("CUDA_VISIBLE_DEVICES")
            if env is None:
                self.mask = "all"
            else:
                want = [x.strip() for x in env.split(",") if x.strip()]
                self.mask = {c.index for c in cards if str(c.index) in want or any(c.uuid.startswith(w) for w in want if len(w) > 3)}
        return None if self.mask == "all" else self.mask

    def account(self, live: dict) -> dict:
        cfg, now, be = self.c, time.time(), self.backend()
        cs = be.cards()
        mask = self.card_mask(cs)
        cards = {c.index: Acct(c.index, c.total_gb, c.used_gb, c.util) for c in cs if mask is None or c.index in mask}
        per_pid = be.usage()
        ours, on_card = set(), {}
        for j in self.jobs("running"):
            mem, g = set(live.get(j["id"], [])), j.get("gpu")
            ours |= mem
            if g not in cards:
                continue
            a, s = cards[g], j["spec"]
            act = sum(v for (cg, p), v in per_pid.items() if cg == g and p in mem)
            j["vram_now"], j["vram_peak"] = round(act, 2), max(float(j.get("vram_peak") or 0), round(act, 2))
            known = B.known_caps(self.hist, s, cfg["hist_min_n"])
            own = B.vram_charge(j, now, known, cfg["trust_measured"], B.hist_max(self.hist, s), cfg["settle_s"])
            j["vram_booked"] = round(max(act, own), 1)
            a.pool_gb += j["vram_booked"]
            a.foreign_gb -= act
            a.jobs += 1
            a.exclusive = a.exclusive or bool(s.get("exclusive"))
            if s.get("tag"):
                a.tags[s["tag"]] = a.tags.get(s["tag"], 0) + 1
            on_card.setdefault(g, []).append(j)
        for g, a in cards.items():
            a.foreign_gb = max(0.0, a.foreign_gb + a.used_gb)
            a.foreign_procs = len({p for (cg, p) in per_pid if cg == g and p not in ours})
        for g, js in on_card.items():
            self.watch_vram(cards[g], js)
        return cards

    def watch_vram(self, a: Acct, js: list) -> None:
        """Jobs measured above their declaration: an event per OVER_STEP_GB of growth; when the card is about to run
        out, the one furthest above its declaration is stopped (it is requeued oom_retries times), not a neighbour
        chosen by whichever allocation fails first."""
        over = []
        for j in js:
            s, peak = j["spec"], float(j.get("vram_peak") or 0)
            if not B.over_declared(s, peak):
                continue
            over.append(j)
            if peak >= float(j.get("vram_over") or 0) + OVER_STEP_GB:
                j["vram_over"] = peak
                self.log("vram_over", id=j["id"], name=s["name"], gpu=a.index, declared=s.get("vram_gb"), peak=peak,
                         card_free_gb=round(a.total_gb - a.used_gb, 1))
                self.status_line(j, "running on card %d: VRAM %.1f GB > %.1f GB declared (--vram)" % (
                    a.index, peak, float(s.get("vram_gb") or 0)))
        free, floor = a.total_gb - a.used_gb, float(self.c.get("over_stop_free_gb") or 0)
        over = [j for j in over if not j.get("stop")]
        if over and floor > 0 and free < floor:
            j = max(over, key=lambda x: float(x.get("vram_now") or 0) - float(x["spec"].get("vram_gb") or 0))
            j["stop"] = "VRAM %.1f GB used > %.1f GB declared with %.1f GB free on card %d" % (
                float(j.get("vram_now") or 0), float(j["spec"].get("vram_gb") or 0), max(free, 0.0), a.index)
            self.log("vram_stop", id=j["id"], name=j["spec"]["name"], gpu=a.index, why=j["stop"])

    def idle_cards(self, cards: dict, now: float) -> list:
        for g, a in cards.items():
            if a.jobs == 0 and a.foreign_gb <= self.c["idle_vram_gb"]:
                self.idle_since.setdefault(g, now)
            else:
                self.idle_since.pop(g, None)
        return [a for g, a in cards.items() if g in self.idle_since and now - self.idle_since[g] >= self.c["idle_s"]]

    # ------------------------------------------------------------------------------------------ admission
    def set_why(self, j: dict, why: str) -> None:
        self.note_wait(j, wait_cause(why))
        if j.get("why") != why:
            j["why"] = why
            self.status_line(j, "queued: " + why)

    def note_wait(self, j: dict, cause: str = "") -> None:
        """Accumulate j["waited"][cause] in seconds, so "why did it wait" has an answer afterwards."""
        now = time.time()
        if j.get("cause"):
            w = j.setdefault("waited", {})
            w[j["cause"]] = round(w.get(j["cause"], 0.0) + now - j.get("t_cause", now), 1)
        j["cause"], j["t_cause"] = cause, now

    def admit(self, cards: dict) -> None:
        cfg, now, trust = self.c, time.time(), self.c["trust_measured"]
        idle = self.idle_cards(cards, now)
        cpu_used, taken, pin_load, young = 0.0, set(), Counter(), 0.0
        for j in self.jobs("running"):
            known = B.known_caps(self.hist, j["spec"], cfg["hist_min_n"]) if trust else {}
            cpu_used += B.cpu_charge(j, now, known.get("cpu"), cfg["young_s"])
            pins = host.parse_cpus(j.get("cpus") or "")
            taken |= set(pins)
            pin_load.update(pins)
            if self.mem["source"] != "budget":       # a budget already counts every running job's booking
                young += B.ram_reserve(j, now, float(j.get("ram_booked") or 0), cfg["young_s"])
        budget = float(cfg["cpu_budget"] or host.cores() * cfg["cpu_overcommit"])
        self.cpu_info = dict(charged=round(cpu_used, 1), budget=round(budget, 1))
        line, used = self.mem["line"], self.mem["used"]
        queue = sorted(self.jobs("queued"), key=lambda j: (-float(j["spec"].get("priority") or 0), j["t_submit"]))
        starts, reserved = Counter(), False
        self.started = 0
        for j in queue:
            s = Spec(**j["spec"])
            st = {a: self.state_of(a) for a in s.after}
            bad = [a for a in s.after if st[a] in ("failed", "cancelled")]
            if bad:
                self.finish(j, "failed", None, "dependency %s %s" % (bad[0], st[bad[0]]))
                continue
            wait = [a for a in s.after if st[a] != "done"]
            gpu_job = B.is_gpu(j["spec"])
            why = ("after %s (%s)" % (wait[0], st[wait[0]]) if wait else
                   "waiting for %s" % s.when_exists if s.when_exists and not Path(s.when_exists).exists() else
                   "start limit this round" if starts[gpu_job] >= cfg["max_starts"] else "")
            if why:
                self.set_why(j, why)
                continue
            known = B.known_caps(self.hist, j["spec"], cfg["hist_min_n"]) if trust else {}
            n_cpu = max(1, min(max(s.cpu, 1), int(known.get("cpu", max(s.cpu, 1)) + 0.999)))
            ram = B.ram_need(j["spec"], known, B.hist_max(self.hist, j["spec"], "ram"), cfg["ram_default_gb"])
            v_need = B.vram_need(j["spec"], known, trust, B.hist_max(self.hist, j["spec"]))
            soft = "CPU %.0f + %d cores > budget %.0f" % (cpu_used, n_cpu, budget) if cpu_used + n_cpu > budget else ""
            hard = ("memory %.0f + %.0f young + %.0f GB > line %.0f GB (%s)" % (used, young, ram, line, self.mem["source"])
                    if line and used + young + ram > line else "")
            if hard or (soft and not gpu_job):
                self.set_why(j, hard or soft)
                continue
            a = None
            if gpu_job:
                pool = [c for c in cards.values() if not s.gpus or c.index in s.gpus]
                top = max((c.total_gb - cfg["headroom_gb"] for c in pool), default=0.0)
                if not pool or (not s.exclusive and v_need > top):
                    self.set_why(j, "no card allowed" if not pool and s.gpus else "no card" if not pool else
                                 "never fits: needs %.1f GB, the largest card offers %.1f (retarget --vram)" % (v_need, top))
                    if not j.get("never"):
                        j["never"] = True
                        self.log("never_fits", id=j["id"], name=s.name, need=v_need, largest=top)
                    continue
                a, reasons = choose(replace(s, vram_gb=v_need), idle if soft else pool, cfg, j["id"], v_need)
                if a is None:
                    self.set_why(j, soft or "; ".join("card %s: %s" % kv for kv in sorted(reasons.items())) or "no card")
                    j.setdefault("t_blocked", now)
                    if not soft and not reserved and now - j["t_blocked"] >= cfg["hold_s"]:
                        reserved = True                # the oldest blocked head job reserves the closest card
                        best = max(pool, key=lambda c: c.avail(cfg["headroom_gb"]))
                        best.reserved_for = j["id"]
                        self.set_why(j, j["why"] + " [reserves card %d]" % best.index)
                    continue
            cpus = ""
            if s.cpu and cfg["pin_cpus"] and host.LINUX:
                free = [c for c in host.affinity() if c not in taken]
                if len(free) >= s.cpu:
                    cpus = host.format_cpus(free[:s.cpu])
                elif gpu_job:                          # never idle a card for want of a free core: share the least-pinned
                    cpus = host.format_cpus(sorted(host.affinity(), key=lambda c: (pin_load[c], c))[:s.cpu])
                else:
                    self.set_why(j, "no %d free cores to pin" % s.cpu)
                    continue
            if soft:
                self.log("admit_idle", id=j["id"], name=s.name, card=a.index, blocked=soft)
            j["vram_booked"], j["ram_booked"] = round(v_need, 1) if gpu_job else 0.0, round(ram, 1)
            if not self.launch(j, s, a.index if a else None, cpus):
                continue
            starts[gpu_job] += 1
            self.started += 1
            if a is not None:
                a.jobs += 1
                a.pool_gb += v_need if not s.exclusive else a.total_gb
                a.exclusive = a.exclusive or s.exclusive
                if s.tag:
                    a.tags[s.tag] = a.tags.get(s.tag, 0) + 1
                if a.reserved_for == j["id"]:
                    a.reserved_for = ""
            pins = host.parse_cpus(cpus)
            taken |= set(pins)
            pin_load.update(pins)
            cpu_used += n_cpu
            young += ram

    def launch(self, j: dict, s: Spec, g, cpus: str) -> bool:
        jid, d, ld = j["id"], self.jdir(j["id"]), self.logdir(j)
        d.mkdir(parents=True, exist_ok=True)
        ld.mkdir(parents=True, exist_ok=True)
        if j["tries"] == 0:
            for x in {d, ld}:
                for f in ("DONE", "ERROR"):
                    if (x / f).exists():
                        (x / f).rename(x / ("%s.%s" % (f, time.strftime("%Y%m%d-%H%M%S"))))
        j["tries"] += 1
        k = j["tries"]
        subs = dict(gpu="" if g is None else g, id=jid, job_dir=ld, try_=k)

        def fill(x):
            x = str(x)
            for key, v in subs.items():
                x = x.replace("{%s}" % key.rstrip("_"), str(v))
            return x
        argv = [SHELL, "-c", fill(s.cmd)] if isinstance(s.cmd, str) else [fill(a) for a in s.cmd]
        env = dict(os.environ)
        env.update(self.hooks.call(self.c, "job_env", j) or {})
        env.update({kk: fill(v) for kk, v in (s.env or {}).items()})
        if s.cpu:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):     # thread pools sized to the cores booked
                if var not in (s.env or {}):
                    env[var] = str(s.cpu)
        env.update(CUDA_VISIBLE_DEVICES="" if g is None else str(g), PINKA_JOB_ID=jid, PINKA_JOB_DIR=str(d),
                   PINKA_LOG_DIR=str(ld), PINKA_GPU="" if g is None else str(g), PINKA_TRY=str(k), PINKA_CPUS=cpus,
                   PINKA_RC=str(d / ("rc.%d" % k)), PINKA_TOKEN="%s/%d/%d" % (jid, k, time.time_ns()),
                   PINKA_HOME=str(self.dir), PYTHONUNBUFFERED="1")
        (d / ("rc.%d" % k)).unlink(missing_ok=True)
        pin = (lambda: os.sched_setaffinity(0, host.parse_cpus(cpus))) if cpus else None
        with (ld / "log.txt").open("ab") as f:
            j["log_off"] = f.tell()
            f.write(("\n==> %s try %d on card %s cpus %s\n$ %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), k,
                     "-" if g is None else g, cpus or "-", shlex.join(argv))).encode())
            f.flush()
            try:
                p = subprocess.Popen(["/bin/sh", "-c", '"$@"; echo $? > "$PINKA_RC"', "sh"] + argv, cwd=s.cwd or None,
                                     env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, preexec_fn=pin)
            except OSError as e:
                f.write(("cannot start: %s\n" % e).encode())
                self.finish(j, "failed", None, "cannot start: %s" % e)
                return False
        host.capture(d / "owned.json", p.pid, token="PINKA_TOKEN=" + env["PINKA_TOKEN"])
        self.children[jid] = p
        for key in ("t_blocked", "vram_over", "vram_peak", "vram_now", "never", "ram_over"):
            j.pop(key, None)
        j.update(state="running", gpu=g, cpus=cpus, t0=time.time(), pid=p.pid, why="")
        self.note_wait(j)
        j.pop("t_cause", None)
        self.status_line(j, "running on card %s (try %d, pid %d)" % ("-" if g is None else g, k, p.pid))
        self.log("launch", id=jid, name=s.name, gpu=g, cpus=cpus, pid=p.pid, try_=k, vram_gb=s.vram_gb,
                 booked_gb=j.get("vram_booked"), waited=j.get("waited"))
        return True

    # ------------------------------------------------------------------------------------------ the loop
    def usage_line(self) -> None:
        """One line a minute in usage.jsonl: per card [util %, GB used, GB booked, jobs], cores, ready queue by cause."""
        now = time.time()
        if now - self.usage_t < USAGE_EVERY_S:
            return
        self.usage_t = now
        q, not_ready = {}, 0
        for j in self.jobs("queued"):
            c = j.get("cause") or "new"
            if c in NOT_READY:
                not_ready += 1
            elif B.is_gpu(j["spec"]):
                q[c] = q.get(c, 0) + 1
        run = self.jobs("running")
        line = dict(t=round(now), cards={g: [a.util, round(a.used_gb, 1), round(a.pool_gb, 1), a.jobs] for g, a in self.last.items()},
                    cpu=[round(sum(float(j.get("cores_now") or 0) for j in run), 1), self.cpu_info.get("charged"),
                         self.cpu_info.get("budget")], q_gpu=q, not_ready=not_ready, mem=[round(self.mem["used"], 1), round(self.mem["line"], 1)])
        with (self.dir / "usage.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")

    def prune(self) -> None:
        """Finished jobs older than keep_days leave state.json (jobs/<id>/job.json keeps them)."""
        now = time.time()
        if now - self.prune_t < PRUNE_EVERY_S:
            return
        self.prune_t = now
        cut = now - float(self.c["keep_days"]) * 86400
        old = [i for i, j in self.st["jobs"].items() if j["state"] in FINAL and j.get("t1", now) < cut]
        for i in old:
            del self.st["jobs"][i]
        if old:
            self.log("prune", n=len(old))

    def save(self) -> None:
        atomic_json(self.dir / "state.json", self.st)
        atomic_json(self.dir / "status.json", dict(
            t=time.time(), pid=os.getpid(), cfg=self.c, cards={g: asdict(a) for g, a in self.last.items()},
            cpu=self.cpu_info, mem=self.mem, idle={g: round(time.time() - t) for g, t in self.idle_since.items()},
            counts={s: len(self.jobs(s)) for s in ("queued", "running", "done", "failed", "cancelled")}))

    def round(self) -> None:
        self.c = load_config(self.dir, self.cfg_over)
        self.ingest()
        live, rows = self.reap(host.processes())
        self.measure(live, rows)
        self.hist = read_json(self.dir / "history.json", {}) or {}
        self.read_memory()
        self.guard_ram()
        cards = self.account(live)
        self.last = cards
        if not self.stopping:
            self.admit(cards)
        lm = float(self.c.get("low_memory_gb") or 0)
        if lm and self.margin < lm and time.time() - self.lowmem_t >= self.c["low_memory_every_s"]:
            self.lowmem_t = time.time()
            self.hooks.call(self.c, "low_memory", self.margin)
        self.usage_line()
        self.prune()
        self.save()

    def poll_s(self) -> float:
        thin = self.margin < 2 * self.ram_stop_gb()
        return min(self.c["poll_s"], self.c["fast_poll_s"]) if thin else self.c["poll_s"]

    def busy(self) -> bool:
        return bool(self.jobs("queued", "running")) or any((self.dir / "inbox").glob("*.json"))

    def run(self, once: bool = False, until_done: bool = False, prepare=None) -> int:
        """The loop. `until_done`: return when nothing is queued or running (foreground mode). `prepare(self)` runs
        once the lock is held, before the first round."""
        with (self.dir / "dispatch.lock").open("a") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SystemExit("another dispatcher owns %s" % self.dir)
            self.st = read_json(self.dir / "state.json", None) or {"jobs": {}}
            if prepare:
                prepare(self)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: setattr(self, "stopping", True))
            self.log("dispatcher_start", pid=os.getpid(), running=[j["id"] for j in self.jobs("running")])
            fails = stuck = 0
            while not self.stopping:
                try:
                    self.round()
                    fails = 0
                except Exception:                      # noqa: BLE001 - a bad round is logged and retried
                    fails += 1
                    self.log("round_error", n=fails, tb=traceback.format_exc()[-2000:])
                    if fails >= 20:
                        return 1
                if once or (until_done and not self.busy()):
                    break
                # Foreground mode: nothing running and nothing started for 3 rounds means the rest can never start
                # (a job larger than any card or budget, an unknown `after`); report instead of waiting forever.
                stuck = stuck + 1 if until_done and not self.jobs("running") and not self.started else 0
                if stuck >= 3:
                    self.log("stuck", queued={j["id"]: j.get("why") for j in self.jobs("queued")})
                    break
                time.sleep(self.poll_s())
            if self.stopping and self.stop_on_exit:
                self.stop_all("runner stopped")
            self.log("dispatcher_stop", running=[j["id"] for j in self.jobs("running")])
        return 0

    def stop_all(self, why: str) -> None:
        """Stop every running job's tree and mark it cancelled (foreground mode on SIGTERM / Ctrl-C)."""
        for j in self.jobs("running"):
            try:
                host.stop(self.jdir(j["id"]) / "owned.json", grace=5.0)
            except (RuntimeError, OSError, ValueError) as e:
                self.log("stop_failed", id=j["id"], err=str(e))
            p = self.children.pop(j["id"], None)
            if p is not None:
                p.poll()
            self.finish(j, "cancelled", None, why)
        self.save()
