"""Shared fixtures: a dispatcher on a temporary state dir with fake cards and fake host memory. Jobs are real short
shell processes, so the process-tree code runs too (Linux: /proc; macOS: ps)."""
import time

import pytest

from pinka import gpu as G
from pinka import jobs as J
from pinka.dispatcher import Dispatcher

FAST = dict(poll_s=0.1, fast_poll_s=0.1, idle_s=0.0, smi_age_s=0.0, hist_min_wall_s=0.0, max_starts=10,
            memory_source="none", pin_cpus=False, cpu_budget=64)


class Pool:
    def __init__(self, tmp_path, cards="1x24", memory=(0.0, 0.0, "none"), **cfg):
        self.dir = tmp_path / "state"
        self.gpu = G.Fake(cards)
        self.memory = memory
        self.d = Dispatcher(self.dir, cfg=dict(FAST, **cfg), gpu=self.gpu, memory_fn=lambda src: self.memory)

    def submit(self, cmd, name="job", **kw):
        kw.setdefault("vram_gb", 0.0)
        return J.submit(cmd, name=name, state=self.dir, **kw)

    def job(self, jid):
        return self.d.st["jobs"][jid]

    def state(self, jid):
        return self.d.st["jobs"][jid]["state"] if jid in self.d.st["jobs"] else "inbox"

    def spin(self, cond, timeout=20.0):
        """Run rounds until cond() holds."""
        end = time.time() + timeout
        while time.time() < end:
            self.d.round()
            if cond():
                return
            time.sleep(0.1)
        raise AssertionError("condition not reached; states %s" % {i: (j["state"], j.get("why")) for i, j in self.d.st["jobs"].items()})

    def settle(self, *ids):
        self.spin(lambda: all(self.state(i) in ("done", "failed", "cancelled") for i in ids))

    def cleanup(self):
        self.d.stopping = True                            # no new starts
        for j in self.d.jobs("running", "queued"):
            J.cancel(j["id"], self.dir)
        if self.d.jobs("running", "queued"):
            self.spin(lambda: not self.d.jobs("running", "queued"))


@pytest.fixture
def make_pool(tmp_path):
    pools = []

    def make(**kw):
        p = Pool(tmp_path / str(len(pools)), **kw)
        pools.append(p)
        return p
    yield make
    for p in pools:
        p.cleanup()
