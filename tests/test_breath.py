import numpy as np, pytest
from api.breath.estimate import estimate_rate
FS = 10
def sine(bpm, dur=60, noise=0.3, seed=1):
    t = np.arange(0, dur, 1 / FS); r = np.random.default_rng(seed)
    return np.sin(2 * np.pi * bpm / 60 * t) + noise * r.standard_normal(len(t))

@pytest.mark.parametrize("bpm", [20, 30, 40, 50, 60])
def test_known_rates(bpm):
    r = estimate_rate(sine(bpm), FS)
    assert r["status"] == "OK" and abs(r["breaths_per_min"] - bpm) <= 2.5

def test_pure_noise_is_unreliable():
    bad = [estimate_rate(np.random.default_rng(s).standard_normal(600), FS)["status"] for s in range(10)]
    assert bad.count("UNRELIABLE") >= 9

def test_short_window_unreliable():
    assert estimate_rate(sine(30, dur=20), FS)["status"] == "UNRELIABLE"

def test_gap_uses_longest_window():
    u = np.ones(600, bool); u[100:140] = False
    r = estimate_rate(sine(30), FS, usable=u)
    assert r["status"] == "OK" and r["window_s"][0] >= 14

def test_result_carries_waveform_for_ui():
    r = estimate_rate(sine(30), FS)
    assert len(r["wave"]["y"]) == 300 and 25 <= len(r["wave"]["peaks"]) <= 31

def test_no_real_motion_is_unreliable_even_if_rhythmic():
    t = np.arange(0, 60, 1 / FS)
    assert estimate_rate(0.005 * np.sin(2 * np.pi * 0.5 * t), FS)["status"] == "UNRELIABLE"   # 30/min but ~0 px
    assert estimate_rate(1e-4 * np.random.default_rng(0).standard_normal(600), FS)["status"] == "UNRELIABLE"
