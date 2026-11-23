"""The client side: a job spec, and submitting, cancelling, retargeting and waiting through files in the state dir.

Clients never edit state.json (the dispatcher is its only writer); they drop files in inbox/, cancel/ and retarget/,
which the dispatcher reads at the start of its next round.
"""
from __future__ import annotations

import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import FINAL, atomic_json, read_json, state_dir


@dataclass
class Spec:
    cmd: object                                   # argv list, or one shell string (run with bash -c)
    name: str = "job"
    key: str = ""                                 # history key (default: the name, trailing numbers stripped)
    owner: str = ""
    cwd: str = ""
    log_dir: str = ""                             # default: <state>/jobs/<id>
    env: dict = field(default_factory=dict)       # set on top of the dispatcher's environment
    vram_gb: float = 0.0                          # peak VRAM on its card; 0 = a CPU job (no card)
    cpu: int = 0                                  # cores; > 0 pins the job to that many free cores (Linux)
    ram_gb: float = 0.0                           # host RAM
    exclusive: bool = False                       # alone on its card
    gpus: list = field(default_factory=list)      # allowed card indices (empty: any)
    tag: str = ""                                 # per-card caps by tag (config tag_caps)
    priority: float = 0.0                         # higher first, then submit order
    after: list = field(default_factory=list)     # job ids that must be done first; a failed one fails this job
    when_exists: str = ""                         # stay queued until this path exists
    tries: int = 1
    timeout_h: float = 0.0

    def check(self) -> "Spec":
        if not self.cmd:
            raise ValueError("empty command")
        if self.vram_gb < 0 or self.cpu < 0 or self.ram_gb < 0:
            raise ValueError("negative resource declaration")
        if self.tries < 1:
            raise ValueError("tries must be >= 1")
        self.after = [a for a in (self.after or []) if a]
        self.gpus = [int(g) for g in self.gpus or []]
        return self


def new_id() -> str:
    return "%s-%s" % (time.strftime("%m%d-%H%M%S"), secrets.token_hex(2))


def submit(cmd, name: str = "job", state=None, **kw) -> str:
    """Queue a job; returns its id. kw: any Spec field."""
    kw.setdefault("owner", os.environ.get("PINKA_JOB_ID") or os.environ.get("USER", ""))
    spec = Spec(cmd=cmd, name=name, **kw).check()
    jid = new_id()
    atomic_json(state_dir(state) / "inbox" / ("%s.json" % jid), dict(id=jid, t_submit=time.time(), spec=asdict(spec)))
    return jid


def cancel(jid: str, state=None) -> None:
    p = state_dir(state) / "cancel" / jid
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stop")


def retarget(jid: str, state=None, gpus=None, vram_gb=None, ram_gb=None) -> None:
    """Change a queued job's allowed cards ([] = any) and / or declared VRAM / RAM; the id stays, so `after` holds."""
    d = {k: v for k, v in (("gpus", gpus), ("vram_gb", vram_gb), ("ram_gb", ram_gb)) if v is not None}
    atomic_json(state_dir(state) / "retarget" / jid, d)


def fanout(cmd, arms, name: str, collect=None, state=None, **kw) -> dict:
    """One job per arm (`{arm}` in cmd, env values and log_dir is replaced; jobs are named <name>-<arm>), plus an
    optional CPU job `collect` after all of them (`{arms}` = every arm). Returns {"arms": {arm: id}, "collect": id}."""
    arms = [str(a) for a in arms]
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("fanout needs distinct arms")

    def fill(x, arm):
        if isinstance(x, (list, tuple)):
            return [fill(y, arm) for y in x]
        return x.replace("{arm}", arm).replace("{arms}", " ".join(arms)) if isinstance(x, str) else x
    ids = {}
    for arm in arms:
        k = dict(kw, env={e: fill(v, arm) for e, v in (kw.get("env") or {}).items()})
        if kw.get("log_dir"):
            k["log_dir"] = fill(kw["log_dir"], arm) if "{arm}" in kw["log_dir"] else str(Path(kw["log_dir"]) / arm)
        ids[arm] = submit(fill(cmd, arm), name="%s-%s" % (name, arm), state=state, **k)
    cid = None
    if collect:
        ck = {k: kw[k] for k in ("owner", "cwd", "env", "priority") if k in kw}
        if kw.get("log_dir") and "{arm}" not in kw["log_dir"]:
            ck["log_dir"] = str(Path(kw["log_dir"]) / "collect")
        cid = submit(fill(collect, ""), name="%s-collect" % name, state=state, after=list(ids.values()), **ck)
    return dict(arms=ids, collect=cid)


# ------------------------------------------------------------------------------------------------ readers
def snapshot(state=None) -> tuple:
    """(state.json, status.json, inbox jobs, seconds since the dispatcher's last round)."""
    d = state_dir(state)
    st = read_json(d / "state.json", None) or {"jobs": {}}
    status = read_json(d / "status.json", None) or {}
    inbox = [x for x in (read_json(p) for p in sorted((d / "inbox").glob("*.json"))) if x]
    age = time.time() - status["t"] if status.get("t") else float("inf")
    return st, status, inbox, age


def find(jid: str, state=None):
    """A job record: from state.json, the inbox, or (after it left state.json) its jobs/<id>/job.json."""
    st, _, inbox, _ = snapshot(state)
    j = st["jobs"].get(jid) or next((dict(x, state="inbox") for x in inbox if x["id"] == jid), None)
    return j or read_json(state_dir(state) / "jobs" / jid / "job.json")


def job_states(ids, state=None) -> dict:
    st, _, inbox, _ = snapshot(state)
    out = {x["id"]: "inbox" for x in inbox}
    out.update({i: j["state"] for i, j in st["jobs"].items()})
    for i in ids:
        if i not in out:
            out[i] = (read_json(state_dir(state) / "jobs" / i / "job.json") or {}).get("state", "unknown")
    return {i: out[i] for i in ids}


def wait(ids, poll_s: float = 30.0, state=None) -> dict:
    """Block until every job is done / failed / cancelled (or unknown); returns id -> state."""
    while True:
        s = job_states(ids, state)
        if all(v in FINAL or v == "unknown" for v in s.values()):
            return s
        time.sleep(poll_s)
