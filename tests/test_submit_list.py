"""`pinka submit --list`: a job list into the shared dispatcher (ids, after by name, prefix, map file, resubmit)."""
import json

import pytest

from pinka import cli
from pinka import jobs as J
from pinka.batch import submit_list


def write(tmp_path, rows, name="jobs.json"):
    p = tmp_path / name
    p.write_text(json.dumps(rows))
    return p


CHAIN = [{"name": "train", "cmd": "echo train", "gpu_gb": 8, "cpus": 2},
         {"name": "eval", "cmd": ["echo", "eval"], "vram_gb": 4, "after": ["train"], "tries": 2},
         {"name": "table", "cmd": "echo table", "after": ["eval"]}]


def test_submit_list_chains_by_name_and_shares_the_pool(make_pool, tmp_path):
    pool = make_pool(cards="1x24")
    other = pool.submit("sleep 0.5", name="other", vram_gb=10)              # a job already in the shared pool
    path = write(tmp_path, CHAIN + [{"name": "late", "cmd": "true", "after": [other]}])
    ids = submit_list(path, state=pool.dir, prefix="exp1-", log_dir=str(tmp_path / "logs"))
    assert set(ids) == {"train", "eval", "table", "late"}
    assert len(set(ids.values())) == 4
    m = json.loads((tmp_path / "jobs.ids.json").read_text())
    assert m["ids"] == ids and m["prefix"] == "exp1-" and m["state_dir"] == str(pool.dir.resolve())
    pool.settle(other, *ids.values())
    j = {n: pool.job(i) for n, i in ids.items()}
    assert all(x["state"] == "done" for x in j.values())
    assert j["eval"]["spec"]["after"] == [ids["train"]] and j["late"]["spec"]["after"] == [other]
    assert j["eval"]["spec"]["name"] == "exp1-eval" and j["eval"]["spec"]["key"] == "eval"
    assert j["train"]["spec"]["cpu"] == 2 and j["eval"]["spec"]["tries"] == 2
    assert j["eval"]["t0"] >= j["train"]["t1"] - 0.5 and j["table"]["t0"] >= j["eval"]["t1"] - 0.5
    assert (tmp_path / "logs" / "table" / "DONE").exists()
    assert J.wait(list(ids.values()), poll_s=0.1, state=pool.dir) == {i: "done" for i in ids.values()}


def test_resubmit_keeps_what_holds(make_pool, tmp_path):
    pool = make_pool(cards="1x24")
    flag = tmp_path / "fixed"
    rows = CHAIN + [{"name": "flaky", "cmd": "test -e %s" % flag}, {"name": "after-flaky", "cmd": "true", "after": ["flaky"]}]
    path = write(tmp_path, rows)
    ids = submit_list(path, state=pool.dir)
    pool.settle(*ids.values())
    assert pool.state(ids["flaky"]) == pool.state(ids["after-flaky"]) == "failed"
    flag.touch()
    rows[2]["cmd"] = "echo table v2"                                         # a changed command
    ids2 = submit_list(write(tmp_path, rows), state=pool.dir)
    kept = {n for n in ids if ids[n] == ids2[n]}
    assert kept == {"train", "eval"}                                         # done, same command and dependencies
    pool.settle(*ids2.values())
    assert all(pool.state(i) == "done" for i in ids2.values())
    ids3 = submit_list(path, state=pool.dir)                                 # nothing changed: nothing submitted
    assert ids3 == ids2 and not list((pool.dir / "inbox").glob("*.json"))
    assert submit_list(path, state=pool.dir, prefix="p-") != ids2            # another prefix is another list


def test_resubmit_replaces_changed_queued_jobs_and_refuses_running_ones(make_pool, tmp_path):
    pool = make_pool(cards="1x24")
    rows = [{"name": "long", "cmd": "sleep 30", "vram_gb": 2}, {"name": "next", "cmd": "echo a", "after": ["long"]}]
    ids = submit_list(write(tmp_path, rows), state=pool.dir)
    pool.spin(lambda: pool.state(ids["long"]) == "running")
    rows[1]["cmd"] = "echo b"
    ids2 = submit_list(write(tmp_path, rows), state=pool.dir)
    assert ids2["long"] == ids["long"] and ids2["next"] != ids["next"]
    pool.spin(lambda: pool.state(ids["next"]) == "cancelled")
    rows[0]["cmd"] = "sleep 31"
    with pytest.raises(ValueError, match="running with a different command"):
        submit_list(write(tmp_path, rows), state=pool.dir)
    rows[0]["cmd"], rows[1]["cmd"] = "sleep 30", "echo c"
    before = json.loads((tmp_path / "jobs.ids.json").read_text())
    dry = submit_list(write(tmp_path, rows), state=pool.dir, dry=True)
    assert dry == {"long": ids["long"], "next": "new:next"}
    assert json.loads((tmp_path / "jobs.ids.json").read_text()) == before      # a dry run writes nothing


def test_list_checks_against_the_pool(tmp_path):
    sd = tmp_path / "s"
    with pytest.raises(ValueError, match="unknown after"):
        submit_list([{"name": "a", "cmd": "true", "after": ["0101-000000-abcdef"]}], state=sd)
    with pytest.raises(ValueError, match="no such job"):
        submit_list([{"name": "a", "cmd": "true"}], state=sd, after=["nope"])
    jid = J.submit("true", state=sd)
    ids = submit_list([{"name": "a", "cmd": "true"}, {"name": "b", "cmd": "true", "after": ["a"]}], state=sd,
                      after=[jid], priority=5)
    a, b = (json.loads((sd / "inbox" / ("%s.json" % ids[n])).read_text())["spec"] for n in "ab")
    assert a["after"] == [jid] and b["after"] == [ids["a"]] and a["priority"] == 5   # --after goes to the roots only


def test_ids_stay_unique_within_one_second(tmp_path):
    sd = tmp_path / "s"
    ids = [J.submit("true", state=sd) for _ in range(300)]
    assert len(set(ids)) == 300 and len(list((sd / "inbox").glob("*.json"))) == 300


def test_cli_submit_list_and_wait(tmp_path, capsys):
    sd = str(tmp_path / "s")
    path = write(tmp_path, CHAIN)
    assert cli.main(["--state-dir", sd, "submit", "--list", str(path), "--prefix", "x-", "--env", "A=1"]) == 0
    lines = [ln.split() for ln in capsys.readouterr().out.splitlines()]
    assert [ln[1:3] for ln in lines] == [["x-train", "new"], ["x-eval", "new"], ["x-table", "new"]]
    m = json.loads((tmp_path / "jobs.ids.json").read_text())
    spec = json.loads((tmp_path / "s" / "inbox" / ("%s.json" % m["ids"]["eval"])).read_text())["spec"]
    assert spec["env"]["A"] == "1" and "PATH" in spec["env"] and spec["after"] == [m["ids"]["train"]]
    for bad in (["--vram", "3"], ["--name", "n"], ["--", "true"]):
        with pytest.raises(SystemExit):
            cli.main(["--state-dir", sd, "submit", "--list", str(path)] + bad)
    with pytest.raises(SystemExit):
        cli.main(["--state-dir", sd, "submit", "--prefix", "p", "--vram", "0", "--", "true"])
    from pinka.dispatcher import Dispatcher
    from pinka import gpu as G
    d = Dispatcher(tmp_path / "s", cfg=dict(poll_s=0.1, idle_s=0.0, memory_source="none", pin_cpus=False,
                                            hist_min_wall_s=0.0), gpu=G.Fake("1x24"))
    for _ in range(100):
        d.round()
        if not d.jobs("queued", "running"):
            break
        __import__("time").sleep(0.1)
    capsys.readouterr()
    assert cli.main(["--state-dir", sd, "wait", "--ids-file", str(tmp_path / "jobs.ids.json"), "--poll", "0.1"]) == 0
    assert capsys.readouterr().out.count("done") == 3
