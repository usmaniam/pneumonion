"""Validation metrics. Pure functions, unit-tested. Small-n honest: every proportion carries a Wilson interval."""
import csv, math, random, statistics as st

def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))

def bootstrap_ci(vals, n=2000, seed=0):
    if len(vals) < 2: return None
    r = random.Random(seed); m = sorted(st.mean(r.choices(vals, k=len(vals))) for _ in range(n))
    return (m[int(.025 * n)], m[int(.975 * n) - 1])

def _flag(v): return str(v).strip().lower() in ("1", "yes", "true", "y")
def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def load_manifest(path, split="test"):
    """Columns: clip, truth_bpm | truth_a+truth_b (two independent counters), scenario, should_refuse, split, subject, age_months."""
    out = []
    for r in csv.DictReader(open(path, newline="")):
        if (r.get("split") or "test").strip() != split: continue
        a, b = _f(r.get("truth_a")), _f(r.get("truth_b"))
        truth = _f(r.get("truth_bpm")) if _f(r.get("truth_bpm")) is not None else (st.mean([a, b]) if a is not None and b is not None else None)
        out.append({"clip": r["clip"], "truth": truth, "rater_diff": abs(a - b) if a is not None and b is not None else None,
                    "scenario": (r.get("scenario") or "normal").strip(), "should_refuse": _flag(r.get("should_refuse")),
                    "subject": (r.get("subject") or "unspecified").strip(), "age_months": int(_f(r.get("age_months")) or 300)})
    return out

def summarize(rows, thresholds=(40, 50), margin=3):
    meas = [r for r in rows if not r["should_refuse"] and r["truth"] is not None]
    ok = [r for r in meas if r["status"] == "OK" and r["pred"] is not None]
    errs = [r["pred"] - r["truth"] for r in ok]
    s = {"n_clips": len(rows), "n_measurable": len(meas), "n_ok": len(ok),
         "coverage": (len(ok), len(meas), wilson(len(ok), len(meas)))}
    if ok:
        ae = [abs(e) for e in errs]; sd = st.stdev(errs) if len(errs) > 1 else 0.0; bias = st.mean(errs)
        s |= {"mae": st.mean(ae), "mae_ci": bootstrap_ci(ae), "max_abs_err": max(ae), "bias": bias, "sd": sd,
              "loa": (bias - 1.96 * sd, bias + 1.96 * sd)}
    s["thresholds"] = {}
    for t in thresholds:
        agree = sum((r["pred"] >= t) == (r["truth"] >= t) for r in ok)
        missed = [r["clip"] for r in ok if r["truth"] >= t > r["pred"]]      # fast breathing called normal: the dangerous error
        false_alarm = [r["clip"] for r in ok if r["truth"] < t <= r["pred"]]
        near = sum(abs(r["truth"] - t) <= margin for r in ok)
        s["thresholds"][t] = {"n": len(ok), "agree": agree, "ci": wilson(agree, len(ok)), "missed": missed,
                              "false_alarm": false_alarm, "near_threshold": near}
    stress = [r for r in rows if r["should_refuse"]]
    refused = [r for r in stress if r["status"] != "OK"]
    s["stress"] = {"n": len(stress), "refused": len(refused), "ci": wilson(len(refused), len(stress)),
                   "false_accepts": [r["clip"] for r in stress if r["status"] == "OK"]}
    by = {}
    for r in rows: by.setdefault(r["scenario"], []).append(r)
    s["by_scenario"] = {k: {"n": len(v), "ok": sum(x["status"] == "OK" for x in v)} for k, v in by.items()}
    s["subjects"] = {k: sum(r["subject"] == k for r in rows) for k in {r["subject"] for r in rows}}
    rd = [r["rater_diff"] for r in rows if r.get("rater_diff") is not None]
    s["rater_mad"] = st.mean(rd) if rd else None
    return s
