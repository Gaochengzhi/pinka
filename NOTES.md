# NOTES

Working notes for the first version (0.1.0, 2026-11-24): what was ported, what was dropped, the decisions taken,
how it was tested, and the questions that are still open.

## Source

pinka is a slimmed port of a private single-node GPU pool. That pool ran about 53,000 jobs a month on a shared
multi-GPU box. It had 3,582 lines in 8 modules: dispatcher, CLI, host probe, page-cache trimmer, disk manager,
process trees, capacity planner, worker profiles. pinka has 1,924 lines in 10 modules (1,517 without blank lines,
comments and docstrings), plus 4 test files.

| Module | Lines | Role |
|:--|--:|:--|
| `dispatcher.py` | ~730 | rounds: ingest, reap, measure, guards, admission, launch, usage, foreground loop |
| `cli.py` | ~300 | argparse front end and the read-only reports |
| `host.py` | ~275 | process table (`/proc`, or `ps` elsewhere), owned trees, cores, memory |
| `booking.py` | ~165 | pure functions: history keys, measured booking of VRAM / CPU / RAM |
| `jobs.py` | ~135 | Spec, submit / cancel / retarget / fanout / wait through spool files |
| `batch.py` | ~95 | `pinka run`: foreground mode for a JSON job list |
| `gpu.py` | ~95 | `nvidia-smi` backend (cached), fake backend, no-GPU backend |
| `config.py` | ~85 | state dir, defaults, JSON helpers |
| `hooks.py` | ~45 | site hooks: job_env, try_end, low_memory |

## Ported (kept)

- Per-job `--vram/--cpu/--ram` declarations; several jobs per card; least-loaded card first, best fit breaks ties.
- Booking from measured peaks: the declaration until settle_s, then 1.2 x own peak + 1 GB (capped by the
  declaration); a history cap after 3 finished runs per key (1.1 x p95 + 1 GB); a declaration that proved too low
  stops being a cap, and its peak goes to history whatever the job's end. History keys come from names with run
  suffixes stripped, or `--key`.
- Submit defaults from history when `--vram/--cpu/--ram` are left out.
- `--after` (a failed or cancelled dependency fails the job), `--tries`, `--timeout-h`, `--priority`,
  `--when-exists`, `--exclusive`, `--gpus`, `--env`, `--cwd`, `--log-dir`, `fanout --collect`, `retarget`,
  `cancel`, `wait`.
- Markers: `jobs/<id>/rc.<try>` written by a `sh` wrapper, so they survive a dispatcher restart; `DONE` (JSON) or
  `ERROR` (reason and log tail); `STATUS` (one line); `log.txt` with a header for every try.
- Requeues beyond `--tries`: CUDA OOM in this try's log (`oom_retries`), a root killed by SIGKILL (`kill_retries`),
  a RAM stop (`ram_retries`, RAM raised to the measured peak).
- VRAM guard (`vram_over` event; `vram_stop` of the job furthest above its declaration on a card that is running
  out). RAM guard (`ram_over`; `ram_stop` under `ram_stop_gb` of margin) and RAM admission (young jobs reserve what
  they have not allocated yet, tapering to zero after 5 minutes).
- Measured CPU charging (declared while young, then 1.2 x the peak over 5 minutes), CPU budget from affinity and the
  cgroup quota, pinning to free cores, and sharing the least-pinned cores rather than leaving a card idle.
- `admit_idle`: a GPU job held back only by the CPU budget may start on a card that has been idle.
- Head-of-queue reservation after `hold_s`, so a large job is not starved.
- Process trees by (pid, start time) plus an environment token; pidfd signalling where available; nothing is killed
  by name pattern; resources are freed only when the whole tree is gone. A restarted dispatcher adopts running jobs.
- `events.jsonl`, `usage.jsonl` (one line a minute) and `usage` / `status` / `history` reports; "why it waited"
  accumulated per cause (`waited` in the launch event).

## Dropped

| Feature | Why |
|:--|:--|
| CARLA servers, port blocks, `--carla/--span`, `capacity.py`, `plan`, `profiles.py`, PID admission | simulator-specific; 0 jobs used them in the last 30 days |
| Page-cache trimmer, temp/compile-cache disk manager, system-disk pruning, box watcher | site-specific: now the `low_memory`, `job_env` and `try_end` hooks |
| The container's 98 % working-set kill-line model, `ws_override`, child-SIGKILL detection from log text | replaced by one generic rule: `ram_frac` x the memory limit (cgroup v2 or host), plus the RAM guard |
| Holds (`hold/holds/unhold`) | 6 uses ever; submit a placeholder job instead |
| `--preflight`, static command check, serial-chain hints | conveniences tied to one team's habits |
| `--pin-strict` and `auto_retarget` (widening `--gpus` onto idle cards) | rare (0.2 %); `--gpus` is now strict |
| `--threads`, `--max-rss`, `--train` | almost unused; `--train` became the generic `--tag` with `tag_caps` |
| Concurrent same-key booking (p99 VRAM of k same-key jobs sharing a card, from `usage.jsonl`) | about 100 lines; per-job measured booking covers most of the gain. Open question 3 |
| NUMA-local core choice | small gain, more probing code; pinning uses free cores in affinity order |
| `drain` cancel mode | tied to one simulator runner |
| Kill-report and memory-high counters in `usage` | specific to the container platform |

## Changed on purpose

- **CPU jobs have no card.** `--vram 0` makes a CPU job, which gets `CUDA_VISIBLE_DEVICES=""`. The old pool parked
  CPU stages on a card with a token VRAM of 0.5.
- **The environment is copied by default.** `submit` stores the submitting shell's environment, so a job sees the
  shell's PATH and conda env; the old pool used the dispatcher's environment unless asked. `--no-copy-env`
  restores the old behaviour. To keep `state.json` from growing with it, finished jobs leave `state.json` after
  `keep_days` (14); their record stays in `jobs/<id>/job.json`, and `show`, `wait` and `--after` still find it.
- **Markers are in one place.** `DONE`/`ERROR` are written to `jobs/<id>/` as well as to a custom `--log-dir`. With
  the old pool, waiters blocked forever on files that only ever appeared in the other folder. The README says
  that `rc.<n>` counts tries and that the exit code is the file's content: this was misread before.
- **"Never fits" is reported.** A declaration larger than any allowed card (total minus headroom) gets that reason
  in `list` and a `never_fits` event, and `submit` warns. The old pool left such a job queued with an ordinary VRAM
  reason, and it held a card reservation. Whole-card jobs should use `--exclusive`.
- **Foreground mode replaces the separate runner.** `pinka run jobs.json` reads the list format of the standalone
  runner (`gpu_gb`, `cpus` are accepted as aliases) and has the same behaviour: `after` by name, resume (skip done
  jobs with an unchanged command, one more try for failed ones, `--retry-failed`), stop leftovers of a killed runner
  first, stop everything on SIGTERM, honour `CUDA_VISIBLE_DEVICES`, pin cores and set `OMP_NUM_THREADS`. It also
  stops when the remaining jobs can never start. Differences from that runner: `tries` defaults to 1 (the runner
  retried once by default; set `"tries": 2`); markers are `jobs.pinka/jobs/<name>/DONE`, not `<name>.done`; the
  room on a card is the measured-booking rule, not `min(total - booked, free - young)`; RAM is a budget only with
  `--ram-gb`, otherwise it is measured.
- `headroom_gb` defaults to 2 (was 4, sized for 80 GB cards); `poll_s` to 10 (was 20); `ram_default_gb` to 1 (was 8).

## Tests

`python -m pytest`: 30 tests in about 10 s on macOS (Python 3.14 and 3.9, through `uv run --no-project --with
pytest`). The tests use the fake GPU backend and real short shell jobs. They cover packing, spreading, measured and
history booking, after chains, failed dependencies, retries with rc/DONE markers, custom log dirs, timeout and cancel
stopping the whole tree, CUDA-OOM requeue, VRAM stop, never fits plus retarget, RAM admission, the RAM guard,
exclusive jobs and tag caps, CPU jobs and the job environment, the `CUDA_VISIBLE_DEVICES` mask, `admit_idle`,
hooks, adopting running jobs after a dispatcher restart, the daemon lock, the foreground list (checks, packing,
chains, resume, never fits), and the CLI (submit, fanout, list). They have not yet been run on Linux: the GPU test
below exercised the Linux paths, but not the pytest suite.

## GPU test (2026-11-24, a 2 x RTX 3090 Linux workstation outside any pool)

Both cards were idle before (15 MiB used, no compute process). Everything ran in a scratch directory
(`~/pinka-scratch`), which was deleted afterwards; the cards were back to 15 MiB with no pinka process left.
pinka ran on the system Python 3.12; the jobs used an existing torch 2.14 + CUDA env, read-only. Each job ran
`alloc.py GB SECONDS`: a `torch.empty` of GB on `cuda`, filled, then held for SECONDS.

**Test 1: daemon, card 0 only** (`config.json`: `{"cards": [0], "poll_s": 3, "settle_s": 30, "hist_min_wall_s": 5}`).

```
pack-a, pack-b, pack-c, pack-d: --vram 10, allocate 3 GB, hold 60-75 s
chain:  --vram 4 --after pack-a, allocate 2 GB
flaky:  --vram 0 --tries 2, fails on the first try (rc 7), succeeds on the second
```

- t+15 s: pack-a and pack-b ran on card 0, each booked at its declared 10 GB. pack-c and pack-d waited ("card 0:
  VRAM 2.0 GB free < 10.0"). nvidia-smi showed 3,370 MiB per process.
- After settle_s, a and b were booked at their measured peak, 1.2 x 3.29 + 1 = 4.9 GB each. pack-c started
  (waited 30 s on VRAM), and three jobs ran on card 0 at once (9.9 GB used, 19.8 booked, 2.2 free). pack-d waited:
  the young pack-c still booked its 10 GB.
- chain started as soon as pack-a ended (waited 75.6 s on `after`). pack-d started 6 s later, when pack-b ended.
- flaky: `rc.1` = 7, a retry event, `rc.2` = 0, done.
- All 6 jobs done; the recorded VRAM peaks were 3.29 GB (pack-*) and 2.29 GB (chain).

**Test 2: foreground mode on card 1** (`CUDA_VISIBLE_DEVICES=1 pinka run jobs.json`). The list used the old
runner's field names: three jobs with `gpu_gb` 6 and `cpus` 1 (3 GB for 20 s), then `after` (after all three), then a
CPU job after that. All three started in the same second on card 1, pinned to cores 0, 1 and 2. `after` and the CPU
job (`CUDA_VISIBLE_DEVICES` empty) followed. The run exited 0 after 28 s with all 5 done. A rerun started nothing and
printed the same table.

Commands: the two scripts were `run.sh` (daemon, `pinka submit` x 6, `pinka list` / `status` at t+15 s and t+50 s,
`pinka wait`, then `events.jsonl` filtered to launch / end / retry) and an inline `pinka run`. Both were deleted with
the scratch directory; their full output is summarised above.

## Open questions for the user

1. **Licence and copyright holder.** The LICENSE file is MIT, marked tentative, with "the pinka contributors" as
   holder. Please confirm, or give the name to use.
2. **Should the release chain move to `pinka run`?** pinka's foreground mode covers the standalone runner (the
   differences are listed above). The frozen release repository was not touched. Should its runner be replaced by a
   pinka dependency in a later release, or stay as it is?
3. **Concurrent same-key booking.** The old pool booked k jobs of one key on a card together, at the measured p99 of
   their combined use, which fits more jobs than per-job peaks do. pinka 0.1 leaves it out. Should it come back
   (about 100 lines), on by default?
4. **Config format.** Config is JSON (`config.json`), because `tomllib` only exists from Python 3.11 and the runtime
   is stdlib-only from 3.9. Is TOML wanted anyway (it would need 3.11+ or an optional dependency)?
5. **pynvml.** Only `nvidia-smi` is implemented. An optional NVML backend would avoid a subprocess call every
   `smi_age_s`. Is it worth adding?
6. **The live box.** pinka has not been run on the shared box, by design: it must not book cards outside the
   running pool. Moving the box from the old pool to pinka would need a cut-over plan: drain the old pool, keep its
   history (the `history.json` format is compatible apart from key names), and port the page-cache trimmer and
   temp-disk logic as hooks.
7. **Name and remote.** No GitHub remote was created. Is "pinka" final, and under which account should it be
   published?
