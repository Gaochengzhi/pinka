"""Host probes: the process table, the process trees jobs own, cores and memory.

Linux reads /proc and cgroup v2. Elsewhere the process table comes from `ps` (enough to run and test pinka on a laptop:
no private-memory split, no environment-token adoption, no CPU pinning).

A job's processes are recorded by (pid, start time) in owned.json; `refresh` adds descendants of live members and, on
Linux, any process carrying the job's PINKA_TOKEN in its environment (a daemon that re-parented itself to init). `stop`
signals only processes whose pid still carries the recorded start time. Nothing is found or killed by name pattern.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import atomic_json

LINUX = sys.platform.startswith("linux")
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
GB = 2 ** 30


# ------------------------------------------------------------------------------------------------ process table
def processes(zombies: bool = False) -> dict:
    """pid -> {pid, ppid, start, cpu_s[, rss_gb]}; zombies are left out unless asked for."""
    return _proc_linux(zombies) if LINUX else _proc_ps(zombies)


def _proc_linux(zombies: bool) -> dict:
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % p) as f:
                s = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if s[0] == "Z" and not zombies:
            continue
        out[int(p)] = dict(pid=int(p), ppid=int(s[1]), start=s[19], cpu_s=(int(s[11]) + int(s[12])) / CLK_TCK)
    return out


def _cputime(s: str) -> float:
    """ps TIME: [[dd-]hh:]mm:ss[.ff] -> seconds."""
    days, _, s = s.rpartition("-")
    t = 0.0
    for part in s.split(":"):
        t = t * 60 + float(part)
    return t + 86400 * int(days or 0)


def _proc_ps(zombies: bool) -> dict:
    try:
        txt = subprocess.run(["ps", "-axo", "pid=,ppid=,stat=,rss=,time=,lstart="], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out = {}
    for ln in txt.splitlines():
        f = ln.split(None, 5)
        if len(f) < 6 or (f[2].startswith("Z") and not zombies):
            continue
        try:
            out[int(f[0])] = dict(pid=int(f[0]), ppid=int(f[1]), start=f[5].strip(), cpu_s=_cputime(f[4]),
                                  rss_gb=int(f[3]) / 2 ** 20)
        except ValueError:
            continue
    return out


def rss_gb(pids, rows: dict) -> float:
    """Private memory (RssAnon + RssShmem) of `pids` in GB on Linux: shared file pages (libraries, mmapped data) are
    not counted, so concurrent jobs reading the same files are not charged for them many times. Elsewhere: RSS."""
    if not LINUX:
        return sum(rows.get(p, {}).get("rss_gb", 0.0) for p in pids)
    kb = 0
    for p in pids:
        try:
            with open("/proc/%d/status" % p) as f:
                for ln in f:
                    if ln.startswith(("RssAnon:", "RssShmem:")):
                        kb += int(ln.split()[1])
        except (OSError, ValueError, IndexError):
            pass
    return kb / 2 ** 20


# ------------------------------------------------------------------------------------------------ owned trees
def identity(row: dict) -> dict:
    return dict(pid=row["pid"], start=row["start"])


def same(rec: dict, rows: dict) -> bool:
    """Is the recorded process still the one running under its pid?"""
    return bool(rec) and rec["pid"] in rows and rows[rec["pid"]]["start"] == rec["start"]


def capture(path: Path, pid: int, token: str = "") -> dict:
    """Record a child we just started (zombie or not: we have not reaped it yet)."""
    for _ in range(5):
        rows = processes(zombies=True)
        if pid in rows:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("cannot see pid %d" % pid)
    me = identity(rows[pid])
    rec = dict(root=me, members=[me], token=token)
    atomic_json(path, rec)
    return rec


def refresh(path: Path, rows: dict) -> tuple:
    """Add the live descendants of live members (and token carriers on Linux); return (record, rows)."""
    rec = json.loads(Path(path).read_text())
    owned = {m["pid"] for m in rec["members"] if same(m, rows)}
    tok = (rec.get("token") or "").encode()
    if tok and LINUX:
        t0 = int(rec["root"]["start"])
        for p, r in rows.items():
            if p not in owned and int(r["start"]) >= t0:
                try:
                    with open("/proc/%d/environ" % p, "rb") as f:
                        if tok in f.read().split(b"\0"):
                            owned.add(p)
                except OSError:
                    pass
    while True:
        new = {p for p, r in rows.items() if r["ppid"] in owned}
        if new <= owned:
            break
        owned |= new
    known = {(m["pid"], m["start"]) for m in rec["members"]}
    added = [identity(rows[p]) for p in owned if (p, rows[p]["start"]) not in known]
    if added:
        rec["members"] += added
        atomic_json(path, rec)
    return rec, rows


def live_members(rec: dict, rows: dict) -> list:
    return [m["pid"] for m in rec["members"] if same(m, rows)]


def _signal(member: dict, sig: int) -> None:
    """Signal one recorded process; through a pidfd where there is one, so a reused pid is never hit."""
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            fd = os.pidfd_open(member["pid"])
        except OSError:
            return
        try:
            if same(member, processes()):
                signal.pidfd_send_signal(fd, sig)
        except OSError:
            pass
        finally:
            os.close(fd)
        return
    try:
        if same(member, processes()):
            os.kill(member["pid"], sig)
    except OSError:
        pass


def stop(path: Path, grace: float = 10.0) -> list:
    """SIGTERM, then SIGKILL after `grace`, every live recorded member (children first). Returns the pids hit; raises
    RuntimeError when some survive SIGKILL (stuck in the kernel)."""
    rec, rows = refresh(path, processes())
    hit = live_members(rec, rows)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        for m in reversed(rec["members"]):
            if same(m, rows):
                _signal(m, sig)
        end = time.monotonic() + wait
        while time.monotonic() < end:
            rows = processes()
            rec, rows = refresh(path, rows)
            if not live_members(rec, rows):
                return hit
            time.sleep(0.2)
    if live_members(rec, processes()):
        raise RuntimeError("owned processes did not exit: %s" % path)
    return hit


# ------------------------------------------------------------------------------------------------ cores, memory
def parse_cpus(spec: str) -> list:
    """'0-3,8' -> [0, 1, 2, 3, 8]."""
    out = []
    for part in (spec or "").replace(" ", "").split(","):
        if part:
            a, _, b = part.partition("-")
            out.extend(range(int(a), int(b or a) + 1))
    return out


def format_cpus(cpus) -> str:
    xs, runs = sorted(set(cpus)), []
    for c in xs:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in runs)


def affinity() -> list:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))


def _cgroup_dir() -> Path:
    """This process's cgroup v2 directory ('/sys/fs/cgroup' in most containers)."""
    try:
        for ln in Path("/proc/self/cgroup").read_text().splitlines():
            if ln.startswith("0::"):
                return Path("/sys/fs/cgroup") / ln[3:].lstrip("/")
    except OSError:
        pass
    return Path("/sys/fs/cgroup")


def _up(d: Path, name: str):
    """The first readable, limited `name` from cgroup dir `d` upwards."""
    while True:
        try:
            v = (d / name).read_text().strip()
            if v and not v.startswith("max"):
                return d, v
        except OSError:
            pass
        if d == d.parent or d == Path("/sys/fs/cgroup"):
            return None, None
        d = d.parent


def cores() -> float:
    """Cores this process may use: the affinity mask, capped by a cgroup CPU quota."""
    n = float(len(affinity()))
    if LINUX:
        _, v = _up(_cgroup_dir(), "cpu.max")
        if v:
            q, period = v.split()[:2]
            n = min(n, int(q) / int(period))
    return n


def memory(source: str = "auto") -> tuple:
    """(limit GB, used GB, source). cgroup: memory.max and memory.current - inactive_file (what a container's memory
    limit acts on); host: MemTotal and MemTotal - MemAvailable. (0, 0, "none") when nothing can be read."""
    if not LINUX or source == "none":
        return 0.0, 0.0, "none"
    if source in ("auto", "cgroup"):
        d, v = _up(_cgroup_dir(), "memory.max")
        if d is not None:
            try:
                cur = int((d / "memory.current").read_text())
                stat = dict(ln.split() for ln in (d / "memory.stat").read_text().splitlines() if ln.count(" ") == 1)
                return int(v) / GB, (cur - int(stat.get("inactive_file", 0))) / GB, "cgroup"
            except (OSError, ValueError):
                pass
        if source == "cgroup":
            return 0.0, 0.0, "none"
    try:
        mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines()}
        return mi["MemTotal"] / 2 ** 20, (mi["MemTotal"] - mi["MemAvailable"]) / 2 ** 20, "host"
    except (OSError, KeyError, ValueError, IndexError):
        return 0.0, 0.0, "none"
