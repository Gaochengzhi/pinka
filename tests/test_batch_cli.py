import io
import json

import pytest

from pinka import cli
from pinka import gpu as G
from pinka.batch import load_list, run_batch

CFG = dict(poll_s=0.1, idle_s=0.0, memory_source="none", pin_cpus=False, hist_min_wall_s=0.0)


def write(tmp_path, rows):
    p = tmp_path / "jobs.json"
    p.write_text(json.dumps(rows))
    return p


def test_list_checks(tmp_path):
    with pytest.raises(ValueError, match="unknown after"):
        load_list(write(tmp_path, [{"name": "a", "cmd": ["true"], "after": ["zz"]}]))
    with pytest.raises(ValueError, match="cycle"):
        load_list(write(tmp_path, [{"name": "a", "cmd": "true", "after": ["b"]}, {"name": "b", "cmd": "true", "after": ["a"]}]))
    with pytest.raises(ValueError, match="unknown field"):
        load_list(write(tmp_path, [{"name": "a", "cmd": "true", "gpu_count": 1}]))
    specs = load_list(write(tmp_path, [{"name": "a", "cmd": ["true"], "gpu_gb": 3, "cpus": 2}]))
    assert (specs[0].vram_gb, specs[0].cpu) == (3, 2)


def test_run_batch_packs_chains_and_resumes(tmp_path):
    flag = tmp_path / "fixed"
    rows = [{"name": "t1", "cmd": "sleep 1", "gpu_gb": 8},
            {"name": "t2", "cmd": "sleep 1", "gpu_gb": 8},
            {"name": "eval", "cmd": "echo eval", "vram_gb": 4, "after": ["t1", "t2"]},
            {"name": "flaky", "cmd": "test -e %s" % flag},
            {"name": "report", "cmd": "echo report", "after": ["flaky"]}]
    path, out, fake = write(tmp_path, rows), io.StringIO(), G.Fake("1x24")
    assert run_batch(path, cfg=CFG, gpu=fake, out=out) == 1
    st = json.loads((tmp_path / "jobs.pinka" / "state.json").read_text())["jobs"]
    assert [st[n]["state"] for n in ("t1", "t2", "eval", "flaky", "report")] == ["done", "done", "done", "failed", "failed"]
    assert st["t1"]["gpu"] == st["t2"]["gpu"] == 0                    # two 8 GB jobs share the 24 GB card
    assert abs(st["t1"]["t0"] - st["t2"]["t0"]) < 1.0
    assert st["eval"]["t0"] >= max(st["t1"]["t1"], st["t2"]["t1"]) - 0.5
    flag.touch()                                                       # the cause is fixed: rerun the same list
    assert run_batch(path, cfg=CFG, gpu=fake, out=out) == 0
    st2 = json.loads((tmp_path / "jobs.pinka" / "state.json").read_text())["jobs"]
    assert st2["t1"]["t0"] == st["t1"]["t0"]                           # done jobs were skipped
    assert st2["flaky"]["state"] == st2["report"]["state"] == "done"
    assert "t1" in out.getvalue() and "5 jobs, 5 done" in out.getvalue()


def test_cli_submit_writes_inbox(tmp_path, capsys):
    sd = str(tmp_path / "s")
    assert cli.main(["--state-dir", sd, "submit", "--name", "train-3", "--vram", "12", "--cpu", "4", "--ram", "16",
                     "--after", "x,y", "--tries", "2", "--env", "A=1", "--", "python", "train.py"]) == 0
    jid = capsys.readouterr().out.strip()
    j = json.loads((tmp_path / "s" / "inbox" / ("%s.json" % jid)).read_text())
    s = j["spec"]
    assert (s["vram_gb"], s["cpu"], s["ram_gb"], s["after"], s["tries"], s["cmd"]) == (12, 4, 16, ["x", "y"], 2, ["python", "train.py"])
    assert s["env"]["A"] == "1" and "PATH" in s["env"]                 # the submitting shell's environment is kept
    with pytest.raises(SystemExit):
        cli.main(["--state-dir", sd, "submit", "--", "true"])          # no --vram and no history


def test_cli_fanout_and_list(tmp_path, capsys):
    sd = str(tmp_path / "s")
    cli.main(["--state-dir", sd, "fanout", "--arms", "a,b", "--vram", "2", "--name", "sweep", "--collect", "echo {arms}",
              "--", "echo {arm}"])
    lines = capsys.readouterr().out.split("\n")
    assert [ln.split()[1] for ln in lines if ln] == ["sweep-a", "sweep-b", "sweep-collect"]
    cli.main(["--state-dir", sd, "list"])
    assert capsys.readouterr().out.count("inbox") == 3


def test_run_batch_reports_jobs_that_can_never_start(tmp_path):
    out = io.StringIO()
    path = write(tmp_path, [{"name": "huge", "cmd": "true", "vram_gb": 40}, {"name": "small", "cmd": "true"}])
    assert run_batch(path, cfg=CFG, gpu=G.Fake("1x24"), out=out) == 1
    assert "never fits" in out.getvalue() and "small" in out.getvalue()


def test_cli_history_shows_shares(tmp_path, capsys):
    from pinka import booking as B
    sd = tmp_path / "s"
    sd.mkdir()
    B.hist_record(sd / "history.json", {"name": "ev-1"}, vram_gb=3.0, cores=1.0)
    (sd / "status.json").write_text(json.dumps(dict(t=0, conc={"ev": dict(share=2.5, n_max=3, by_n={})})))
    cli.main(["--state-dir", str(sd), "history"])
    out = capsys.readouterr().out
    assert "ev" in out and "2.5" in out and "3.0/3.0" in out
