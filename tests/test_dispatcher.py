import json
import os

from pinka import booking as B
from pinka import jobs as J
from pinka.dispatcher import Dispatcher

SLEEP = "sleep 30"


def test_packs_jobs_onto_one_card_by_declared_vram(make_pool):
    p = make_pool(cards="1x24")                          # 22 GB after the 2 GB headroom
    ids = [p.submit(SLEEP, name="pack", vram_gb=6) for _ in range(4)]
    p.spin(lambda: sum(p.state(i) == "running" for i in ids) == 3)
    p.d.round()
    waiting = [i for i in ids if p.state(i) == "queued"]
    assert len(waiting) == 1 and "VRAM 4.0 GB free < 6.0" in p.job(waiting[0])["why"]
    assert {p.job(i).get("gpu") for i in ids if p.state(i) == "running"} == {0}


def test_spreads_over_cards_first(make_pool):
    p = make_pool(cards="2x24")
    a, b = p.submit(SLEEP, vram_gb=4), p.submit(SLEEP, vram_gb=4)
    p.spin(lambda: p.state(a) == p.state(b) == "running")
    assert {p.job(a)["gpu"], p.job(b)["gpu"]} == {0, 1}


def test_booking_falls_to_measured_peak(make_pool):
    p = make_pool(cards="1x24", settle_s=0.0)
    a = p.submit(SLEEP, name="big", vram_gb=20)
    p.spin(lambda: p.state(a) == "running")
    p.gpu.pids[p.job(a)["pid"]] = (0, 2.0)               # it only ever uses 2 GB
    b = p.submit(SLEEP, name="other", vram_gb=15)
    p.spin(lambda: p.state(b) == "running")
    assert p.job(a)["vram_booked"] == 3.4                # 1.2 x 2 + 1


def test_declaration_is_kept_while_young(make_pool):
    p = make_pool(cards="1x24", settle_s=600.0)
    a = p.submit(SLEEP, name="big", vram_gb=20)
    p.spin(lambda: p.state(a) == "running")
    p.gpu.pids[p.job(a)["pid"]] = (0, 2.0)
    b = p.submit(SLEEP, name="other", vram_gb=15)
    for _ in range(3):
        p.d.round()
    assert p.state(b) == "queued" and p.job(a)["vram_booked"] == 20.0


def test_history_books_queued_job_below_declaration(make_pool):
    p = make_pool(cards="1x24")
    for _ in range(3):
        B.hist_record(p.dir / "history.json", {"name": "ev"}, vram_gb=2.0)
    a = p.submit(SLEEP, name="ev-1", vram_gb=20)
    b = p.submit(SLEEP, name="ev-2", vram_gb=20)
    p.spin(lambda: p.state(a) == p.state(b) == "running")
    assert p.job(b)["vram_booked"] == 3.2                # 1.1 x p95 2.0 + 1


def test_after_chain_and_failed_dependency(make_pool):
    p = make_pool()
    ok = p.submit("true", name="ok")
    bad = p.submit("exit 3", name="bad")
    c1 = p.submit("true", name="c1", after=[ok])
    c2 = p.submit("true", name="c2", after=[bad])
    p.settle(ok, bad, c1, c2)
    assert p.state(c1) == "done" and p.job(c1)["t0"] >= p.job(ok)["t1"] - 1
    assert p.state(c2) == "failed" and "dependency" in p.job(c2)["why"]
    assert p.job(bad)["rc"] == 3


def test_retry_then_success_markers(make_pool, tmp_path):
    p = make_pool()
    flag = tmp_path / "flag"
    a = p.submit("if [ -e %s ]; then exit 0; else touch %s; exit 1; fi" % (flag, flag), tries=2)
    p.settle(a)
    d = p.dir / "jobs" / a
    assert p.state(a) == "done" and p.job(a)["tries"] == 2
    assert (d / "rc.1").read_text().strip() == "1" and (d / "rc.2").read_text().strip() == "0"
    assert json.loads((d / "DONE").read_text())["state"] == "done"


def test_custom_log_dir_gets_markers_too(make_pool, tmp_path):
    p = make_pool()
    a = p.submit("echo hello", log_dir=str(tmp_path / "logs"))
    p.settle(a)
    assert (tmp_path / "logs" / "DONE").exists() and (p.dir / "jobs" / a / "DONE").exists()
    assert "hello" in (tmp_path / "logs" / "log.txt").read_text()
    b = p.submit("exit 2", log_dir=str(tmp_path / "logs2"))
    p.settle(b)
    assert (tmp_path / "logs2" / "ERROR").exists() and (p.dir / "jobs" / b / "ERROR").exists()


def test_timeout_and_cancel_stop_the_tree(make_pool):
    p = make_pool()
    t = p.submit("sleep 30 & sleep 30; wait", timeout_h=1.5 / 3600)
    c = p.submit(SLEEP)
    p.spin(lambda: p.state(c) == "running")
    J.cancel(c, p.dir)
    p.settle(t, c)
    assert p.state(t) == "failed" and "timeout" in p.job(t)["why"]
    assert p.state(c) == "cancelled"
    from pinka import host
    rows = host.processes()
    for jid in (t, c):
        rec = json.loads((p.dir / "jobs" / jid / "owned.json").read_text())
        assert len(rec["members"]) >= 2 if jid == t else True
        assert not host.live_members(rec, rows)


def test_cuda_oom_in_log_is_retried_once(make_pool):
    p = make_pool()
    a = p.submit("echo 'RuntimeError: CUDA out of memory. Tried to allocate'; exit 1", vram_gb=4)
    p.settle(a)
    j = p.job(a)
    assert j["state"] == "failed" and j["oom_tries"] == 1 and j["tries"] == 2 and "out of memory" in j["why"]


def test_vram_stop_picks_the_over_declared_job(make_pool):
    p = make_pool(cards="1x24")
    good, greedy = p.submit(SLEEP, name="good", vram_gb=4), p.submit(SLEEP, name="greedy", vram_gb=4)
    p.spin(lambda: p.state(good) == p.state(greedy) == "running")
    p.gpu.pids[p.job(good)["pid"]] = (0, 4.0)
    p.gpu.pids[p.job(greedy)["pid"]] = (0, 19.5)       # 23.5 of 24 GB used: under 1 GB free
    p.spin(lambda: p.job(greedy).get("oom_tries") == 1)
    assert p.state(good) == "running"
    assert B.hist_max(json.loads((p.dir / "history.json").read_text()), {"name": "greedy"}) == 19.5


def test_never_fits_then_retarget(make_pool):
    p = make_pool(cards="1x24")
    a = p.submit("true", vram_gb=30)
    p.spin(lambda: "never fits" in (p.d.st["jobs"].get(a) or {}).get("why", ""))
    J.retarget(a, p.dir, vram_gb=6)
    p.settle(a)
    assert p.state(a) == "done"


def test_ram_line_blocks_start(make_pool):
    p = make_pool(memory=(100.0, 85.0, "host"))          # line 90 GB
    a = p.submit("true", ram_gb=10)
    p.spin(lambda: p.state(a) == "queued" and "memory" in p.job(a).get("why", ""))
    p.memory = (100.0, 50.0, "host")
    p.settle(a)
    assert p.state(a) == "done"


def test_ram_guard_stops_the_job_above_its_declaration(make_pool, monkeypatch):
    from pinka import dispatcher as D
    p = make_pool(memory=(100.0, 80.0, "host"), ram_stop_gb=15.0)   # line 90 GB, margin 10 < 15: the guard acts
    a = p.submit(SLEEP, name="hog", ram_gb=2)
    b = p.submit(SLEEP, name="calm", ram_gb=2)
    p.spin(lambda: p.state(a) == p.state(b) == "running")
    monkeypatch.setattr(D.host, "rss_gb", lambda pids, rows: 20.0 if p.job(a).get("pid") in pids else 0.5)
    p.spin(lambda: p.job(a).get("ram_tries") == 1)
    assert p.job(a)["spec"]["ram_gb"] == 21.0 and p.state(b) == "running"


def test_exclusive_and_tag_caps(make_pool):
    p = make_pool(cards="1x24", tag_caps={"train": 1})
    e = p.submit(SLEEP, exclusive=True)
    s = p.submit(SLEEP, vram_gb=1)
    p.spin(lambda: p.state(e) == "running")
    p.d.round()
    assert p.state(s) == "queued" and "exclusive" in p.job(s)["why"]
    J.cancel(e, p.dir)
    t1, t2 = p.submit(SLEEP, vram_gb=1, tag="train"), p.submit(SLEEP, vram_gb=1, tag="train")
    p.spin(lambda: p.state(t1) == "running" and p.state(s) == "running")
    assert p.state(t2) == "queued" and "tag train" in p.job(t2)["why"]


def test_cpu_job_gets_no_card_and_env(make_pool, tmp_path):
    p = make_pool(cards="1x24")
    out = tmp_path / "env"
    a = p.submit("env > %s" % out, cpu=3, env={"FOO": "{id}"})
    p.settle(a)
    env = dict(ln.split("=", 1) for ln in out.read_text().splitlines() if "=" in ln)
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["FOO"] == a and env["OMP_NUM_THREADS"] == "3"
    assert env["PINKA_JOB_ID"] == a and env["PINKA_RC"].endswith("rc.1")


def test_card_mask_from_cuda_visible_devices(make_pool, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    p = make_pool(cards="2x24")
    a = p.submit(SLEEP, vram_gb=4)
    p.spin(lambda: p.state(a) == "running")
    assert p.job(a)["gpu"] == 1 and set(p.d.last) == {1}


def test_cpu_budget_relaxed_on_idle_card_only(make_pool):
    p = make_pool(cards="1x24", cpu_budget=1)
    g = p.submit(SLEEP, vram_gb=2, cpu=4)
    c = p.submit(SLEEP, cpu=4)
    p.spin(lambda: p.state(g) == "running")
    p.d.round()
    assert p.state(c) == "queued" and "CPU" in p.job(c)["why"]
    assert any('"admit_idle"' in ln for ln in (p.dir / "events.jsonl").read_text().splitlines())


def test_hooks(make_pool, tmp_path):
    hook = tmp_path / "site_hooks.py"
    hook.write_text("import pathlib\n"
                    "def job_env(job):\n    return {'SITE_TMP': '/scratch/' + job['id']}\n"
                    "def try_end(job, rc):\n    pathlib.Path(%r).write_text(str(rc))\n" % str(tmp_path / "ended"))
    p = make_pool(hooks={"job_env": "%s:job_env" % hook, "try_end": "%s:try_end" % hook})
    a = p.submit("echo $SITE_TMP")
    p.settle(a)
    assert ("/scratch/" + a) in (p.dir / "jobs" / a / "log.txt").read_text()
    assert (tmp_path / "ended").read_text() == "0"


def test_new_dispatcher_adopts_running_jobs(make_pool):
    p = make_pool()
    a = p.submit("sleep 1.5")
    p.spin(lambda: p.state(a) == "running")
    p.d = Dispatcher(p.dir, cfg=p.d.cfg_over, gpu=p.gpu, memory_fn=p.d.memory_fn)   # a restart: no Popen handles
    p.settle(a)
    assert p.state(a) == "done" and p.job(a)["rc"] == 0


def test_daemon_lock_and_once(tmp_path):
    d = Dispatcher(tmp_path, cfg=dict(gpu_backend="none", memory_source="none"))
    assert d.run(once=True) == 0
    assert json.loads((tmp_path / "status.json").read_text())["counts"]["queued"] == 0
    assert os.path.exists(tmp_path / "dispatch.lock")
