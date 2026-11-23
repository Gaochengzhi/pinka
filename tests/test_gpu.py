import sys
import types
from types import SimpleNamespace as NS

import pytest

from pinka import gpu as G

GB = 2 ** 30


def fake_pynvml():
    m = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    def util(h):
        if h == 1:
            raise NVMLError("not supported")
        return NS(gpu=40)
    m.NVMLError = NVMLError
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetCount = lambda: 2
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetMemoryInfo = lambda h: NS(total=24 * GB, used=(3 if h == 0 else 0.5) * GB)
    m.nvmlDeviceGetUUID = lambda h: b"GPU-%d" % h
    m.nvmlDeviceGetUtilizationRates = util
    m.nvmlDeviceGetComputeRunningProcesses = lambda h: (
        [NS(pid=123, usedGpuMemory=3 * GB)] if h == 0 else [NS(pid=9, usedGpuMemory=None)])
    return m


def test_nvml_backend_reads_cards_and_processes():
    be = G.Nvml(max_age=0.0, nvml=fake_pynvml())
    cards = be.cards()
    assert [(c.index, c.uuid, c.total_gb, c.used_gb, c.util) for c in cards] == [
        (0, "GPU-0", 24.0, 3.0, 40), (1, "GPU-1", 24.0, 0.5, 0)]
    assert be.usage() == {(0, 123): 3.0, (1, 9): 0.0}


def test_backend_auto_detection(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml())
    assert isinstance(G.backend("auto"), G.Nvml)
    assert isinstance(G.backend("nvidia-smi"), G.NvidiaSmi)
    monkeypatch.setitem(sys.modules, "pynvml", None)                    # not installed
    assert isinstance(G.backend("auto"), G.NvidiaSmi)
    with pytest.raises(ImportError):
        G.backend("nvml")
    assert isinstance(G.backend("fake:2x8"), G.Fake) and isinstance(G.backend("none"), G.NoGpu)


def test_nvidia_smi_parsing(monkeypatch):
    out = {"gpu=index,uuid,memory.used,memory.total,utilization.gpu": ["0, GPU-a, 1024, 24576, 7", "1, GPU-b, 0, 24576, [N/A]"],
           "compute-apps=gpu_uuid,pid,used_memory": ["GPU-a, 42, 1024", "GPU-b, 43, [N/A]"]}
    be = G.NvidiaSmi(max_age=0.0)
    monkeypatch.setattr(be, "_query", lambda q: out[q])
    assert [(c.index, c.used_gb, c.util) for c in be.cards()] == [(0, 1.0, 7), (1, 0.0, 0)]
    assert be.usage() == {(0, 42): 1.0, (1, 43): 0.0}
