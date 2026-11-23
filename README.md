# pinka

Pack GPU jobs onto shared cards by measured memory. pinka is a job pool for one machine. Each job declares its
peak VRAM, cores and RAM. A dispatcher starts every job on a card with room, so several small jobs share one card.
Once a job has run long enough, or its name has a history, it is booked at the VRAM it actually used rather than at
what it declared. Jobs can depend on each other (`--after`) and be retried (`--tries`). Every job gets an exit-code
file and a DONE or ERROR marker, and the host's RAM is guarded. Python 3.9+, standard library only, no root, no
database: the state is a directory of JSON files.

The name is Mandarin *pīn kǎ* (拼卡): sharing a card, the way people share a taxi.

## Why another scheduler

Most single-node tools hand out whole GPUs. task-spooler (`ts -G`) and simple_gpu_scheduler give each job whole free
cards, and pueue does not know about GPUs at all. Ray can put several tasks on one GPU (`num_gpus=0.25`), but you
declare the fractions by hand, it never measures what tasks really use, and Ray is a large dependency. Slurm with GRES
shards needs root and cluster setup. pinka fits one case: many jobs with uneven, roughly known memory needs (evaluation
episodes, small trainings, simulators) packed onto a few large cards by their measured footprint, with a spool any
script, agent or shell can write to. If your jobs each need a whole GPU, use task-spooler or pueue.

## Install

```bash
pip install .            # from a checkout; installs the `pinka` command
pinka --help
```

Linux with NVIDIA drivers (`nvidia-smi` on `PATH`) for GPU jobs. On macOS and other systems without NVIDIA cards
pinka still runs CPU-only jobs, and the tests run there with a fake GPU backend.

## Five-minute quickstart

```bash
# 1. Start the dispatcher (keep it running: tmux, screen, or a systemd user unit).
pinka daemon &

# 2. Queue jobs. --vram is the peak VRAM in GB (0 = a CPU-only job).
A=$(pinka submit --name eval-1 --vram 6 --cpu 2 --ram 8 -- python eval.py --shard 1)
B=$(pinka submit --name eval-2 --vram 6 --cpu 2 --ram 8 -- python eval.py --shard 2)
pinka submit --name report --vram 0 --after $A,$B -- python report.py
pinka submit --name train --vram 20 --tries 3 --timeout-h 12 -- python train.py

# 3. Watch.
pinka list                 # queue: state, card, booked/declared VRAM, why a job waits
pinka status               # per card: used, booked, outside the pool, free; which jobs run where
pinka show $A              # the job record and the tail of its log

# 4. Block until jobs end (exit 0 when all are done), e.g. from a script.
pinka wait $A $B
```

Each job runs as `bash -c CMD` (one string) or as an argv list, in the submitting shell's directory and with its
environment (`--no-copy-env` uses the dispatcher's environment instead). Its output goes to
`~/.pinka/jobs/<id>/log.txt`, or to `--log-dir`.

### Foreground mode: run a list here

```bash
pinka run jobs.json [--gpus 0,1] [--cpus 16] [--ram-gb 64] [--retry-failed]
```

`jobs.json` is a list of job objects with any submit field; `after` names other jobs in the list:

```json
[{"name": "train", "cmd": ["python", "train.py"], "vram_gb": 40, "cpu": 8, "ram_gb": 32},
 {"name": "eval-a", "cmd": "python eval.py --part a", "vram_gb": 6, "after": ["train"], "tries": 2},
 {"name": "eval-b", "cmd": "python eval.py --part b", "vram_gb": 6, "after": ["train"], "tries": 2},
 {"name": "table", "cmd": "python table.py", "after": ["eval-a", "eval-b"]}]
```

The command runs the same dispatcher in the foreground, in a state directory of its own (`jobs.pinka/` next to the
list). It exits when every job has ended: 0 if all are done, 1 otherwise, after printing a table of the jobs. It also
exits, and reports why, when the remaining jobs can never start (one needs more VRAM than any card has, say).
Rerunning the same list skips jobs that are already done with an unchanged command. A job that failed gets one more
try, or all of its tries with `--retry-failed`. Anything a killed runner left running is stopped first. Ctrl-C or
SIGTERM stops every job. `CUDA_VISIBLE_DEVICES` on the runner (or `--gpus`) limits which cards it uses, so pinka can
work inside a reservation that another scheduler gave you.

## CLI reference

All commands accept `--state-dir DIR` before the subcommand (default: `$PINKA_HOME`, else `$XDG_STATE_HOME/pinka`,
else `~/.pinka`).

| Command | What it does |
|:--|:--|
| `daemon [--once]` | the dispatcher loop (one per state dir; a second one exits) |
| `submit [flags] -- CMD ...` | queue a job, print its id |
| `fanout --arms a,b,c [--collect CMD] [flags] -- CMD` | one job per arm (`{arm}` in CMD, `--env` values and `--log-dir`), jobs named NAME-ARM, plus an optional CPU job after all of them (`{arms}` = every arm) |
| `run JOBS.json [--gpus] [--cpus] [--ram-gb] [--retry-failed]` | foreground mode (above) |
| `list [--all]` | running and queued jobs, and jobs that ended in the last 6 hours |
| `show ID` | full job record and log tail |
| `cancel ID ...` | cancel queued jobs or stop running ones |
| `retarget ID ... [--vram GB] [--ram GB] [--gpus 0,2]` | change a queued job's declaration or allowed cards; its id stays, so `--after` chains hold |
| `wait ID ... [--poll S]` | block until the jobs end; exit 0 if all are done |
| `status` | cards (used, booked by the pool, used outside it, free), CPU and RAM budget, idle cards next to waiting jobs |
| `usage [--hours H]` | card-hours split into computing, job running with GPU idle, queued (by cause), no work |
| `history [--all]` | per history key: runs, VRAM p95/max, cores, RAM |
| `config` | the effective configuration |

Submit flags: `--name` (default: the command's first word), `--key` (history key), `--vram GB`, `--cpu N`,
`--ram GB`, `--exclusive` (whole card), `--gpus 0,2` (allowed cards), `--tag T` (per-card caps), `--priority P`
(higher first), `--after ID,ID`, `--when-exists PATH` (stay queued until the file exists), `--tries N`,
`--timeout-h H`, `--env K=V`, `--cwd`, `--log-dir`, `--owner`, `--no-copy-env`. If `--vram`, `--cpu` or `--ram` is
left out and the name has history, the default is p95 x 1.2 of earlier runs. A job with no history must declare
`--vram`.

Python API: `pinka.submit(cmd, name=..., vram_gb=..., after=[...]) -> id`, `pinka.wait(ids)`, `pinka.cancel(id)`,
`pinka.retarget(id, vram_gb=...)`, `pinka.fanout(...)`.

The job's environment includes `CUDA_VISIBLE_DEVICES` (its card; empty for a CPU job), `PINKA_JOB_ID`, `PINKA_GPU`,
`PINKA_TRY`, `PINKA_CPUS`, `PINKA_JOB_DIR`, `PINKA_LOG_DIR` and `PINKA_HOME`. A job with `--cpu N` also gets
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to N unless its own env sets them. In the command and env values,
`{gpu}`, `{id}`, `{job_dir}` and `{try}` are replaced with the job's values.

## How packing and booking work

Each round (`poll_s`, default 10 s), the dispatcher checks queued jobs in priority order, then in submit order. A job
whose `after` jobs are not done, or whose `--when-exists` file is missing, waits. A job whose `after` job failed or was
cancelled fails too. Then:

1. **Booking.** A job is booked at its declaration, at least at first. Its "history key" is its name with trailing
   run numbers removed, so `eval-s3` and `eval-12` share the key `eval`. Once 3 finished runs under a key exist, a
   new job is booked at `min(declared, 1.1 x p95 of their peaks + 1 GB)`. Once a running job is older than
   `settle_s` (600 s), its booking becomes `min(declared, max(history cap, 1.2 x its own peak + 1 GB))`. A running
   job always counts at least what it uses right now (from `nvidia-smi`, per process, summed over its process tree).
   A declaration that proved too low stops being a cap. A job measured above `1.1 x declared + 1 GB` is booked at
   `1.2 x its peak + 1 GB`, and its peak is written to history even if it fails. The next job under that key is then
   placed by what was measured.
2. **Fitting a card.** A card has room for a job when `total - headroom - (VRAM used outside the pool) - (pool
   bookings) >= need`. Of the cards with room, the least loaded wins (fewest jobs), and best fit breaks ties.
   Spreading keeps every card busy; best fit leaves room for a large job. `--exclusive` needs an empty card and
   takes all of it, and `tag_caps` limits how many jobs with a given tag share a card.
3. **CPU.** The CPU budget is `cpu_budget`, else the cores this process may use (affinity and cgroup quota) times
   `cpu_overcommit`. A job is charged its declared cores for its first `young_s` (300 s), then 1.2 x its measured
   peak over the last 5 minutes. On Linux, `--cpu N` pins the job to N free cores. A GPU job blocked only by the CPU
   budget still starts on a card that has been idle for `idle_s`, so a card never waits on CPU bookkeeping.
4. **RAM.** A job starts only if `used + RAM still to come from young jobs + its RAM <= ram_frac x limit`. The limit
   is the cgroup v2 `memory.max` if one is set, else host RAM; "used" excludes reclaimable page cache. With
   `ram_budget_gb` (or `pinka run --ram-gb`), RAM is booked against that budget instead of being measured. Each job's
   memory is its tree's private memory (RssAnon + RssShmem), so files mapped by many jobs are not charged many times.
5. **No starvation.** A job blocked at the head of the queue for `hold_s` (15 min) reserves the card closest to
   fitting it. Lower-priority jobs stop starting there until it starts.
6. **Never fits.** A job that needs more than any allowed card offers says so in `list` ("never fits"), and `submit`
   warns. Fix it with `pinka retarget ID --vram GB`, or use `--exclusive` for a whole card.

The dispatcher also watches running jobs:

- **VRAM over declaration**: an event, and a line in the job's `STATUS`. If the card has less than
  `over_stop_free_gb` free, the job furthest above its declaration is stopped and requeued (`oom_retries`), instead
  of a neighbour being hit by whichever allocation fails first.
- **CUDA out of memory** in this try's log: requeued `oom_retries` times beyond `--tries`.
- **SIGKILL from outside** (the kernel OOM killer, rc 137): requeued `kill_retries` times beyond `--tries`.
- **RAM over declaration**: an event. When the margin under the RAM line falls below `ram_stop_gb` (default 4 % of
  the limit), the job furthest above its declaration is stopped and requeued once, with its RAM declaration raised
  to what it measured. The kernel would otherwise kill the largest process, often a well-behaved training job.
- **Timeout** (`--timeout-h`), **cancel**: the whole process tree is stopped (SIGTERM, then SIGKILL).

Process trees are tracked by (pid, start time). A process re-parented to init is still found through its parent
chain or, on Linux, through a token in its environment. Nothing is ever found or killed by name pattern. Resources
are released only after the whole tree has exited. Restarting the dispatcher does not touch running jobs: the new
one adopts them from `state.json`.

### Advice that kept cards full in practice

- Split large runs into many small jobs sized near their measured need, not one job per card. The tail then spreads
  over all free cards.
- Make GPU jobs only drive the GPU. Run scoring, gates and clean-up as separate CPU jobs (`--vram 0`) chained
  `--after` the GPU job they read. No GPU job should wait on a CPU job.
- When one batch waits on another, point `--after` at the other batch's last GPU jobs, not at its CPU read-outs.
- Declare a whole-card job as `--exclusive`. A card's total minus headroom is less than its nominal size, so a job
  declaring the full size would never fit.

## State layout

```
$PINKA_HOME/
  config.json        optional overrides of the defaults (pinka config prints the result); re-read every round
  inbox/<id>.json    submitted jobs, not yet read by the dispatcher
  cancel/<id>        cancel requests        retarget/<id>   retarget requests
  state.json         every job: spec, state, card, tries, bookings, measurements, why it waits (dispatcher only)
  status.json        last round: cards, CPU, RAM, counts, heartbeat (read by list/status)
  history.json       per history key: the last 20 VRAM / cores / RAM peaks
  events.jsonl       one JSON line per event: submit, launch, end, retry, oom, kill, vram_over, vram_stop,
                     ram_over, ram_stop, never_fits, admit_idle, hook_error, ...
  usage.jsonl        one line a minute: per card [util %, GB used, GB booked, jobs], cores, ready queue by cause
  dispatch.lock      one dispatcher per state dir
  jobs/<id>/
    spec.json        the job as submitted
    owned.json       its process tree, by (pid, start time)
    rc.<try>         exit code of try <try> (the file name counts tries; the content is the exit code)
    DONE | ERROR     final marker: DONE is JSON, ERROR has the reason and the log tail
    job.json         the final record (finished jobs leave state.json after keep_days)
    log.txt, STATUS  unless --log-dir is given; the log dir also gets its own DONE / ERROR
```

To wait for a job from a shell without pinka, block on `jobs/<id>/DONE` or `ERROR`. These appear in the job folder
whether or not `--log-dir` was given.

The spool has no authentication. Anyone who can write to the state directory can submit jobs that run as the
dispatcher's user. Keep it private (`chmod 700`), or run one dispatcher per user.

## Hooks

Machine-specific behaviour goes into hooks, Python functions named in `config.json`:

```json
{"hooks": {"job_env": "/opt/site/pinka_site.py:job_env",
           "try_end": "/opt/site/pinka_site.py:try_end",
           "low_memory": "mysite.hooks:drop_page_cache"},
 "low_memory_gb": 64}
```

| Hook | Called | Use |
|:--|:--|:--|
| `job_env(job) -> dict` | before each try | extra environment, e.g. a per-job `TMPDIR` on a large disk, shared compile caches |
| `try_end(job, rc)` | after each try, once its whole tree has exited | clean up that `TMPDIR` |
| `low_memory(margin_gb)` | while the margin under the RAM line is below `low_memory_gb`, at most every `low_memory_every_s` | drop page cache, free a RAM disk |

A hook that raises is logged as `hook_error` and otherwise ignored.

## Configuration

`pinka config` prints every key and its value. The main ones: `poll_s`, `headroom_gb` (2), `cards` (allowed card
indices), `gpu_backend` (`nvidia-smi`, `none`, `fake:2x24`), `settle_s`, `trust_measured` (false: always book the
declaration), `cpu_budget`, `cpu_overcommit`, `pin_cpus`, `memory_source` (`auto`, `cgroup`, `host`, `none`),
`ram_budget_gb`, `ram_frac`, `ram_stop_gb`, `oom_retries`, `kill_retries`, `ram_retries`, `tag_caps`, `keep_days`,
`hooks`.

## Limitations

- One machine. There is no network protocol; several users can share a dispatcher only through a shared directory.
- NVIDIA only (via `nvidia-smi`). VRAM per process comes from `nvidia-smi --query-compute-apps`. In containers whose
  PID namespace hides host PIDs, the per-job numbers can be missing; card totals still work.
- The dispatcher polls (default every 10 s): a job starts up to one round after it becomes ready.
- MIG and MPS are not modelled. A card is one memory pool.
- CPU pinning, private-memory measurement and environment-token adoption are Linux-only. macOS uses `ps` and is
  meant for development and tests.
- Measured booking trusts that a job's past peak predicts its future one. A job whose memory spikes late should
  declare honestly and set `trust_measured` false, or use a distinct `--key`.

## Development

```bash
python -m pytest           # about 10 s; fake GPUs, real short shell jobs; Linux or macOS
```

## Licence

MIT (tentative; see LICENSE).
