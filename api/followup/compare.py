"""Follow-up comparison rules, scheduling, and calendar file. No model calls."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml

VSEV = {"CONTINUE": 0, "IMPROVING": 0, "NO_VIDEO_VERDICT": 1, "NOT_IMPROVING": 2, "WORSE": 3, "URGENT": 4}
TEXT = {
    "URGENT": "Seek care now.", "WORSE": "See a health worker today.",
    "NOT_IMPROVING": "Not improving. See a health worker.",
    "NO_VIDEO_VERDICT": "No video verdict. Count manually or retry.",
    "IMPROVING": "Improving. Continue care, recheck as scheduled, return if worse.",
    "CONTINUE": "No change of concern. Recheck as scheduled, return if worse."}

def load_fu(path=None):
    return yaml.safe_load(Path(path or Path(__file__).parents[1] / "rules" / "who_followup.yaml").read_text())

def _ok(c):
    return c["status"] == "OK" and c.get("bpm") is not None

def compare(cur, prev, first, day, fu=None):
    """cur/prev/first: classify() output with inputs{bpm,status}. Most severe verdict wins."""
    fu = fu or load_fu()
    hits = []
    new_signs = set(cur["danger_present"]) - set(prev["danger_present"])
    if new_signs or cur["severity"] == 4:
        hits.append("URGENT")
    both = _ok(cur["inputs"]) and _ok(prev["inputs"])
    if cur["severity"] > prev["severity"]:
        hits.append("WORSE")
    if both and prev["inputs"]["bpm"] < prev["threshold_bpm"] <= cur["inputs"]["bpm"]:
        hits.append("WORSE")
    if not _ok(cur["inputs"]):
        hits.append("NO_VIDEO_VERDICT")
    elif day >= fu["not_improving_day"]:
        b, thr = cur["inputs"]["bpm"], cur["threshold_bpm"]
        fb = first["inputs"]["bpm"] if _ok(first["inputs"]) else None
        first_fast = fb is not None and fb >= first["threshold_bpm"]
        if b >= thr or (first_fast and b > fb - fu["not_improving_margin_bpm"]):
            hits.append("NOT_IMPROVING")
    if not hits:
        fb = first["inputs"]["bpm"] if _ok(first["inputs"]) else None
        better = cur["severity"] == 1 and fb is not None and cur["inputs"]["bpm"] < fb
        hits.append("IMPROVING" if better else "CONTINUE")
    v = max(hits, key=lambda h: VSEV[h])
    return {"verdict": v, "text": TEXT[v], "triggered": hits, "new_danger_signs": sorted(new_signs)}

def schedule(created_at, fu=None):
    fu = fu or load_fu()
    return [created_at + timedelta(hours=h) for h in fu["schedule_hours"]]

def overdue(created_at, n_rechecks_done, now, fu=None):
    """-> (next_due or None, is_overdue). Drives the in-app banner."""
    s = schedule(created_at, fu)
    if n_rechecks_done >= len(s):
        return None, False
    return s[n_rechecks_done], now >= s[n_rechecks_done]

def make_ics(case_id, created_at, now=None, fu=None):
    now = now or datetime.now(timezone.utc)
    f = lambda d: d.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev = "".join(
        f"BEGIN:VEVENT\r\nUID:{case_id}-r{i}@pneumonion\r\nDTSTAMP:{f(now)}\r\nDTSTART:{f(d)}\r\n"
        "SUMMARY:Pneumonion recheck: count your child's breathing\r\n"
        "DESCRIPTION:Open Pneumonion and tap Recheck. If your child gets worse\\, go to a health facility now.\r\n"
        "END:VEVENT\r\n" for i, d in enumerate(schedule(created_at, fu), 1))
    return f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//Pneumonion//EN\r\n{ev}END:VCALENDAR\r\n"
