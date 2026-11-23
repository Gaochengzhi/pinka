"""GPU backends: what cards exist, how much VRAM each uses, and how much each process uses.

`Nvml` uses the optional pynvml binding (package nvidia-ml-py): no subprocess per query. `NvidiaSmi` calls
nvidia-smi and needs no Python binding. Both cache their answers for `max_age` seconds: every query takes the driver's
device lock, and a busy host should not pay for it many times a second. `gpu_backend` "auto" (the default) picks
Nvml when pynvml imports and initialises, else nvidia-smi; "nvml" or "nvidia-smi" forces one. `Fake` serves made-up
cards for tests and dry runs ("fake:2x24").

Card indices are the driver's PCI-bus order (nvidia-smi's); jobs get CUDA_DEVICE_ORDER=PCI_BUS_ID so that CUDA
numbers the cards the same way.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class Card:
    index: int
    uuid: str
    total_gb: float
    used_gb: float
    util: int = 0


class NvidiaSmi:
    def __init__(self, max_age: float = 10.0):
        self.max_age, self.cache = max_age, {}

    def _query(self, q: str) -> list:
        t, lines = self.cache.get(q, (0.0, None))
        if lines is None or time.time() - t > self.max_age:
            try:
                lines = subprocess.run(["nvidia-smi", "--query-" + q, "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
            except (OSError, subprocess.TimeoutExpired):
                lines = []
            self.cache[q] = (time.time(), lines)
        return lines

    def cards(self) -> list:
        out = []
        for ln in self._query("gpu=index,uuid,memory.used,memory.total,utilization.gpu"):
            f = [x.strip() for x in ln.split(",")]
            if len(f) < 5 or not f[0].isdigit():
                continue
            try:
                out.append(Card(int(f[0]), f[1], float(f[3]) / 1024, float(f[2]) / 1024, int(f[4]) if f[4].isdigit() else 0))
            except ValueError:
                continue
        return sorted(out, key=lambda c: c.index)

    def usage(self) -> dict:
        """{(card index, pid): GB} for every compute process."""
        idx = {c.uuid: c.index for c in self.cards()}
        out = {}
        for ln in self._query("compute-apps=gpu_uuid,pid,used_memory"):
            f = [x.strip() for x in ln.split(",")]
            if len(f) == 3 and f[0] in idx and f[1].isdigit():
                try:
                    out[(idx[f[0]], int(f[1]))] = float(f[2]) / 1024
                except ValueError:
                    out[(idx[f[0]], int(f[1]))] = 0.0
        return out


class Nvml:
    def __init__(self, max_age: float = 10.0, nvml=None):
        if nvml is None:
            import pynvml as nvml                         # optional dependency: pip install nvidia-ml-py
        nvml.nvmlInit()
        self.n, self.max_age, self.t, self.snap = nvml, max_age, 0.0, ([], {})

    def _read(self) -> tuple:
        if time.time() - self.t <= self.max_age:
            return self.snap
        n, cards, use = self.n, [], {}
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            uuid = n.nvmlDeviceGetUUID(h)
            try:                                          # v2: `used` without driver-reserved memory, as nvidia-smi
                mem = n.nvmlDeviceGetMemoryInfo(h, version=n.nvmlMemory_v2)
            except Exception:                             # noqa: BLE001 - older bindings or drivers: v1
                mem = n.nvmlDeviceGetMemoryInfo(h)
            try:
                util = int(n.nvmlDeviceGetUtilizationRates(h).gpu)
            except n.NVMLError:
                util = 0
            cards.append(Card(i, uuid.decode() if isinstance(uuid, bytes) else uuid, mem.total / 2 ** 30,
                              mem.used / 2 ** 30, util))
            try:
                procs = n.nvmlDeviceGetComputeRunningProcesses(h)
            except n.NVMLError:
                procs = []
            for p in procs:
                use[(i, int(p.pid))] = (p.usedGpuMemory or 0) / 2 ** 30   # None when the driver cannot tell
        self.t, self.snap = time.time(), (cards, use)
        return self.snap

    def cards(self) -> list:
        return self._read()[0]

    def usage(self) -> dict:
        return self._read()[1]


class Fake:
    """Cards from a spec like '2x24' (two 24 GB cards) or '24,48'. Tests set `pids` ({pid: (card, GB)}) and `other`
    ({card: GB used outside any process we know})."""

    def __init__(self, spec: str = "1x24"):
        n, _, gb = spec.partition("x")
        sizes = [float(gb)] * int(n) if gb else [float(x) for x in spec.split(",")]
        self.sizes, self.pids, self.other, self.util = sizes, {}, {}, {}

    def cards(self) -> list:
        used = dict(self.other)
        for g, gb in self.pids.values():
            used[g] = used.get(g, 0.0) + gb
        return [Card(i, "FAKE-%d" % i, t, used.get(i, 0.0), self.util.get(i, 0)) for i, t in enumerate(self.sizes)]

    def usage(self) -> dict:
        return {(g, p): gb for p, (g, gb) in self.pids.items()}


class NoGpu:
    def cards(self) -> list:
        return []

    def usage(self) -> dict:
        return {}


def backend(name: str, max_age: float = 10.0):
    """"auto" (pynvml if it imports and initialises, else nvidia-smi), "nvml", "nvidia-smi", "none" or "fake:SPEC"."""
    if name.startswith("fake:"):
        return Fake(name[5:])
    if name == "none":
        return NoGpu()
    if name in ("auto", "nvml"):
        try:
            return Nvml(max_age)
        except Exception:                                 # noqa: BLE001 - ImportError, NVMLError (no driver), ...
            if name == "nvml":
                raise
    return NvidiaSmi(max_age)
