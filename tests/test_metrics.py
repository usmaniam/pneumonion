import csv, pytest
from eval.metrics import wilson, summarize, load_manifest, bootstrap_ci

def R(truth, pred, status="OK", refuse=False, scen="normal", clip="c"):
    return {"clip": clip, "truth": truth, "pred": pred, "status": status, "should_refuse": refuse, "scenario": scen, "subject": "adult", "rater_diff": None}

def test_wilson_known_values():
    lo, hi = wilson(0, 10); assert lo == 0 and hi == pytest.approx(0.278, abs=.002)
    lo, hi = wilson(10, 10); assert hi == 1 and lo == pytest.approx(0.722, abs=.002)
    assert wilson(0, 0) is None

def test_mae_bias_loa():
    s = summarize([R(30, 32), R(40, 38), R(50, 53)])
    assert s["mae"] == pytest.approx(7 / 3) and s["bias"] == pytest.approx(1.0) and s["max_abs_err"] == 3
    assert s["loa"][0] < s["bias"] < s["loa"][1]

def test_unreliable_excluded_from_mae_but_counted_in_coverage():
    s = summarize([R(30, 31), R(40, None, "UNRELIABLE")])
    assert s["mae"] == 1 and s["coverage"][:2] == (1, 2)

def test_threshold_missed_fast_breathing_is_separate_from_false_alarm():
    s = summarize([R(52, 48, clip="miss"), R(45, 51, clip="alarm"), R(30, 31), R(60, 58)])
    t50 = s["thresholds"][50]
    assert t50["missed"] == ["miss"] and t50["false_alarm"] == ["alarm"] and t50["agree"] == 2
    assert s["thresholds"][40]["missed"] == []

def test_stress_refusal_and_false_accepts():
    s = summarize([R(None, None, "UNRELIABLE", True, clip="a"), R(None, 33, "OK", True, clip="b"), R(None, None, "NO_TRACK", True, clip="c")])
    assert s["stress"]["refused"] == 2 and s["stress"]["false_accepts"] == ["b"] and s["n_measurable"] == 0

def test_stress_clips_never_pollute_accuracy():
    assert summarize([R(30, 30), R(None, 99, "OK", True)])["mae"] == 0

def test_manifest_two_counters_and_split(tmp_path):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "truth_a", "truth_b", "truth_bpm", "split", "should_refuse"])
        w.writerow(["a.mp4", 40, 44, "", "test", "no"]); w.writerow(["b.mp4", "", "", 30, "dev", "no"]); w.writerow(["c.mp4", "", "", "", "test", "yes"])
    rows = load_manifest(p, "test")
    assert [r["clip"] for r in rows] == ["a.mp4", "c.mp4"] and rows[0]["truth"] == 42 and rows[0]["rater_diff"] == 4 and rows[1]["should_refuse"]
    assert [r["clip"] for r in load_manifest(p, "dev")] == ["b.mp4"]

def test_bootstrap_deterministic():
    assert bootstrap_ci([1, 2, 3, 4]) == bootstrap_ci([1, 2, 3, 4]) and bootstrap_ci([1]) is None
