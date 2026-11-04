"""Deterministic WHO-style classifier. No model calls."""
from pathlib import Path
import yaml

DANGER = ["unable_to_drink", "vomits_everything", "convulsions", "sleepy", "stridor", "blue_lips"]
SEV = {"NO_SIGNS_FOUND": 1, "NO_CLASSIFICATION": 1, "FAST_BREATHING": 2,
       "ASSESS_TODAY": 3, "URGENT_REFERRAL": 4}

def load_cfg(path=None):
    return yaml.safe_load(Path(path or Path(__file__).with_name("who_imci.yaml")).read_text())

def threshold(age_months, cfg):
    t = cfg["fast_breathing_bpm"]
    return t["young_infant"] if age_months < 2 else t["infant_2_11"] if age_months < 12 else t["child_12_59"]

def classify(age_months, bpm, status, danger, chest_indrawing, spo2=None, cfg=None):
    cfg = cfg or load_cfg()
    if not 0 <= age_months <= 59:
        raise ValueError("age_months must be 0-59")
    missing = [k for k in DANGER if k not in danger]
    if missing:  # unanswered items must be re-asked, never assumed "no"
        raise ValueError(f"unanswered danger items: {missing}")
    present = [k for k in DANGER if danger[k]]
    if spo2 is not None and spo2 < cfg["spo2_severe_below"]:
        present.append("low_spo2")
    reliable = status == "OK" and bpm is not None
    thr = threshold(age_months, cfg)
    fired = []  # (classification, rule_id), in precedence-tie order
    if age_months < 2:
        fired.append(("URGENT_REFERRAL", "R1_YOUNG_INFANT"))
    if present:
        fired.append(("URGENT_REFERRAL", "R2_DANGER_SIGN"))
    if chest_indrawing:
        fired.append((cfg["chest_indrawing_classification"], "R3_CHEST_INDRAWING"))
    if reliable and bpm >= thr:
        fired.append(("FAST_BREATHING", "R4_FAST_BREATHING"))
    if not reliable:
        fired.append(("NO_CLASSIFICATION", "R6_UNRELIABLE"))
    elif not fired:
        fired.append(("NO_SIGNS_FOUND", "R5_NO_SIGNS"))
    cls, rule = max(fired, key=lambda f: SEV[f[0]])  # severe wins; first listed wins ties
    return {
        "classification": cls, "severity": SEV[cls], "rule_id": rule,
        "fired_rules": [r for _, r in fired], "action_phrase": cfg["actions"][cls],
        "threshold_bpm": thr, "danger_present": present,
        "repeat_count": bool(reliable and abs(bpm - thr) <= cfg["borderline_margin_bpm"]),
        "inputs": {"age_months": age_months, "bpm": bpm, "status": status,
                   "chest_indrawing": chest_indrawing, "spo2": spo2},
        "rules_version": cfg["version"],
    }
