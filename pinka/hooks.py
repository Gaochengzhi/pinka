"""Optional hooks for machine-specific behaviour, named in config.json:

  "hooks": {"job_env": "mypkg.site:job_env", "try_end": "/opt/site/pinka_hooks.py:try_end", "low_memory": ...}

  job_env(job) -> dict      extra environment for a try about to start (e.g. a per-job TMPDIR on a large disk)
  try_end(job, rc)          a try ended and its whole process tree is gone (e.g. remove that TMPDIR)
  low_memory(margin_gb)     the margin under the RAM line is below low_memory_gb (e.g. drop page cache); at most every
                            low_memory_every_s

A hook that raises is logged (event hook_error) and otherwise ignored.
"""
from __future__ import annotations

import importlib
import importlib.util
import traceback


def load(ref: str):
    mod, _, fn = ref.rpartition(":")
    if mod.endswith(".py"):
        spec = importlib.util.spec_from_file_location("pinka_hook_%d" % abs(hash(mod)), mod)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn)


class Hooks:
    def __init__(self, log):
        self.log, self.cache = log, {}

    def call(self, cfg: dict, name: str, *args):
        ref = (cfg.get("hooks") or {}).get(name)
        if not ref:
            return None
        try:
            if ref not in self.cache:
                self.cache[ref] = load(ref)
            return self.cache[ref](*args)
        except Exception:                               # noqa: BLE001 - a site hook never stops the dispatcher
            self.log("hook_error", hook=name, ref=ref, tb=traceback.format_exc()[-1500:])
            return None
