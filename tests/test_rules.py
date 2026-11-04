import pytest
from api.rules.engine import classify, DANGER
NO = {k: False for k in DANGER}
def c(age, bpm, status="OK", danger=None, ind=False, spo2=None):
    return classify(age, bpm, status, {**NO, **(danger or {})}, ind, spo2)

@pytest.mark.parametrize("age,bpm,exp", [
    (2, 49, "NO_SIGNS_FOUND"), (2, 50, "FAST_BREATHING"), (11, 49, "NO_SIGNS_FOUND"), (11, 50, "FAST_BREATHING"),
    (12, 39, "NO_SIGNS_FOUND"), (12, 40, "FAST_BREATHING"), (59, 39, "NO_SIGNS_FOUND"), (59, 40, "FAST_BREATHING")])
def test_thresholds(age, bpm, exp):
    assert c(age, bpm)["classification"] == exp

@pytest.mark.parametrize("bpm", [10, 59, 60])
def test_under_2_months_always_urgent(bpm):
    assert c(1, bpm)["classification"] == "URGENT_REFERRAL"

def test_danger_beats_fast_breathing():
    assert c(24, 55, danger={"blue_lips": True})["rule_id"] == "R2_DANGER_SIGN"

def test_spo2_severe():
    assert c(24, 30, spo2=89)["classification"] == "URGENT_REFERRAL"
    assert c(24, 30, spo2=90)["classification"] == "NO_SIGNS_FOUND"

def test_indrawing_beats_fast_breathing():
    assert c(24, 55, ind=True)["classification"] == "ASSESS_TODAY"

def test_unreliable_never_says_no_signs():
    r = c(24, None, status="UNRELIABLE")
    assert r["classification"] == "NO_CLASSIFICATION"

def test_unreliable_still_surfaces_danger():
    assert c(24, None, "UNRELIABLE", {"convulsions": True})["classification"] == "URGENT_REFERRAL"

def test_borderline_repeat():
    assert c(24, 42)["repeat_count"] and not c(24, 30)["repeat_count"]

def test_unanswered_item_rejected():
    with pytest.raises(ValueError):
        classify(24, 30, "OK", {"stridor": False}, False)

def test_audit_fields():
    r = c(24, 45)
    assert r["rules_version"] and r["fired_rules"] and r["inputs"]["age_months"] == 24
