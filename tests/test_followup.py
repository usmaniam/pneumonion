from datetime import datetime, timedelta, timezone
from api.rules.engine import classify, DANGER
from api.followup.compare import compare, overdue, make_ics
NO = {k: False for k in DANGER}
def c(bpm, status="OK", danger=None, ind=False, age=24):
    return classify(age, bpm, status, {**NO, **(danger or {})}, ind)
FIRST = c(52)

def test_new_danger_sign_urgent():
    assert compare(c(45, danger={"sleepy": True}), c(45), FIRST, 1)["verdict"] == "URGENT"
def test_worse_than_previous():
    assert compare(c(45, ind=True), c(45), FIRST, 1)["verdict"] == "WORSE"
def test_threshold_crossing():
    assert compare(c(44), c(36), c(36), 1)["verdict"] == "WORSE"
def test_day3_not_improving_above_threshold():
    assert compare(c(48), c(48), FIRST, 3)["verdict"] == "NOT_IMPROVING"
def test_day3_not_improving_margin():
    assert compare(c(38), c(38), c(40), 3)["verdict"] == "NOT_IMPROVING"
def test_improving():
    assert compare(c(34), c(45), FIRST, 2)["verdict"] == "IMPROVING"
def test_unreliable_no_video_verdict():
    assert compare(c(None, "UNRELIABLE"), c(45), FIRST, 1)["verdict"] == "NO_VIDEO_VERDICT"
def test_severe_wins_over_unreliable():
    assert compare(c(None, "UNRELIABLE", {"stridor": True}), c(45), FIRST, 1)["verdict"] == "URGENT"
def test_overdue_banner_with_clock_override():
    t0 = datetime(2026, 11, 3, tzinfo=timezone.utc)
    assert overdue(t0, 0, t0 + timedelta(hours=23))[1] is False
    assert overdue(t0, 0, t0 + timedelta(hours=25))[1] is True
    assert overdue(t0, 3, t0 + timedelta(days=9)) == (None, False)
def test_ics_has_three_events():
    t0 = datetime(2026, 11, 3, tzinfo=timezone.utc)
    assert make_ics("abc", t0).count("BEGIN:VEVENT") == 3
