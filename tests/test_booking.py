from pinka import booking as B


def test_hist_prefix_strips_run_suffixes():
    assert B.hist_prefix("Eval-s3") == "eval"
    assert B.hist_prefix("eval-k12_t0") == "eval"
    assert B.hist_prefix("train2") == "train"
    assert B.hist_prefix("job") == "job"
    assert B.hist_key({"name": "a-1", "key": "shared-7"}) == "shared"


def test_known_caps_need_min_runs_and_use_p95():
    hist = {"eval": dict(vram=[2.0, 2.0], cores=[1.0, 1.0], ram=[3.0, 3.0])}
    assert B.known_caps(hist, {"name": "eval-1"}) == {}
    hist["eval"] = dict(vram=[2.0, 2.0, 4.0], cores=[1.0, 2.0, 1.0], ram=[3.0, 3.0, 3.0])
    caps = B.known_caps(hist, {"name": "eval-1"})
    assert caps == {"vram_gb": round(4.0 * 1.1 + 1, 1), "cpu": 2.4, "ram_gb": 4.6}


def test_vram_need_and_charge():
    spec = dict(name="x", vram_gb=20.0)
    assert B.vram_need(spec, {}) == 20.0
    assert B.vram_need(spec, {"vram_gb": 3.2}) == 3.2
    assert B.vram_need(spec, {"vram_gb": 3.2}, trust=False) == 20.0
    assert B.vram_need(spec, {}, seen=30.0) == 31.0            # a recorded peak above the declaration wins
    j = dict(spec=spec, t0=0.0, vram_peak=2.0)
    assert B.vram_charge(j, now=10.0, settle_s=600) == 20.0     # young, no history: the declaration
    assert B.vram_charge(j, now=1000.0, settle_s=600) == 3.4    # settled: 1.2 x peak + 1
    j["vram_peak"] = 30.0
    assert B.vram_charge(j, now=1000.0) == 37.0                 # broke the declaration: no longer capped by it


def test_cpu_and_ram():
    j = dict(spec=dict(cpu=8), t0=0.0, cores_peak=2.0)
    assert B.cpu_charge(j, now=10.0) == 8.0
    assert B.cpu_charge(j, now=1000.0) == 2.4
    spec = dict(ram_gb=10.0)
    assert B.ram_need(spec, {}, 0.0, 1.0) == 10.0
    assert B.ram_need(spec, {"ram_gb": 4.0}, 0.0, 1.0) == 4.0
    assert B.ram_need(spec, {}, 20.0, 1.0) == 21.0
    assert B.ram_need({}, {}, 0.0, 1.0) == 1.0
    assert B.ram_reserve(dict(t0=0.0, rss_now=2.0), now=150.0, need=10.0, young_s=300) == 4.0


def test_conc_shares_learns_from_card_minutes():
    jobs = [dict(spec=dict(name="ev-%d" % i, vram_gb=10), gpu=0, t0=1.0, t1=10000.0, state="done") for i in range(2)]
    jobs.append(dict(spec=dict(name="other", vram_gb=10), gpu=1, t0=1.0, state="running"))
    lines = [dict(t=100 + 60 * k, cards={"0": [50, 8.0 + (k == 0), 20.0, 2], "1": [50, 5.0, 10.0, 1]}) for k in range(40)]
    out = B.conc_shares(lines, jobs, now=5000.0, min_n=30)
    assert list(out) == ["ev"] and out["ev"]["n_max"] == 2 and out["ev"]["share"] == 4.5   # p99 of 40 samples: the top one
    assert B.conc_shares(lines, jobs, now=5000.0, min_n=41) == {}          # too few card-minutes to trust


def test_conc_group_and_add():
    info = dict(share=3.0, n_max=3)
    assert B.conc_group(1, 0.0, 10.0, 10.0, 3.0, 1.1) == 10.0              # one job: its own booking
    assert B.conc_group(4, 0.0, 40.0, 10.0, 3.0, 1.1) == 13.2              # four: 4 x 1.1 x 3
    assert B.conc_group(4, 20.0, 40.0, 10.0, 3.0, 1.1) == 20.0             # never below what they use now
    g = dict(m=3, act=0.0, own_sum=30.0, own_max=10.0, booked=10.0)
    assert B.conc_add(g, info, 10.0, 1.1) == 3.2                           # the fourth adds the group's growth
    assert B.conc_add(dict(g, m=4), info, 10.0, 1.1) == 10.0               # beyond n_max + 1: its own booking
    assert not B.conc_eligible(dict(name="ev", vram_gb=10, exclusive=True), {"ev": info})
    assert not B.conc_eligible(dict(name="ev", vram_gb=10), {"ev": info}, seen=30.0)
