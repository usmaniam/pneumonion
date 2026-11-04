"""Run the full measurement pipeline on a manifest of clips and write results.csv, report.md, bland_altman.png.
  python -m eval.run_eval --manifest eval/manifest.csv --clips eval/clips --tracks api|cache|box
'api' calls SAM (about $0.12 a clip) and caches tracks; 'cache' replays them free; 'box' uses a fixed central ROI (no SAM)."""
import argparse, asyncio, csv, pickle, tempfile
from pathlib import Path
import cv2, numpy as np
from api.video.transcode import transcode, VideoError
from api.meta.sam import FrameTrack
from api.breath.pipeline import measure
from .metrics import load_manifest, summarize

def box_tracks(mp4):
    cap = cv2.VideoCapture(str(mp4)); n, W, H = int(cap.get(7)), int(cap.get(3)), int(cap.get(4)); cap.release()
    b = (int(.2 * W), int(.1 * H), int(.8 * W), int(.95 * H))
    ras = np.ones((b[3] - b[1], b[2] - b[0]), bool)
    return {i: FrameTrack(i, b, ras, 1.0) for i in range(n)}

async def get_tracks(mode, mp4, clip, age, cache):
    cp = Path(cache) / (Path(clip).stem + ".pkl")
    if mode == "box": return box_tracks(mp4)
    if mode == "cache":
        return pickle.load(open(cp, "rb")) if cp.exists() else None    # own cache files only: never unpickle untrusted input
    from api.meta.sam import track_video
    from api.meta.client import shared_client
    tracks, _ = await track_video(shared_client(), mp4, age)
    cp.parent.mkdir(parents=True, exist_ok=True); pickle.dump(tracks, open(cp, "wb"))
    return tracks

async def run(a):
    rows, tmp = [], Path(tempfile.mkdtemp(prefix="pn_eval_"))
    for m in load_manifest(a.manifest, a.split):
        mp4 = tmp / "c.mp4"; r = {**m, "pred": None, "status": "REJECTED_VIDEO", "quality": None, "agreement": None}
        try:
            info = transcode(Path(a.clips) / m["clip"], mp4, max_s=70)
            tracks = await get_tracks(a.tracks, mp4, m["clip"], m["age_months"], a.cache)
            if not tracks or len(tracks) < 0.5 * info["duration"] * 10:
                r["status"] = "NO_TRACK"          # the app would ask for a retake
            else:
                res = measure(mp4, tracks)
                r.update(pred=res["breaths_per_min"], status=res["status"], quality=res["quality"], agreement=res["method_agreement_bpm"])
        except (VideoError, FileNotFoundError) as e:
            r["status"] = "REJECTED_VIDEO"; print(m["clip"], type(e).__name__)
        rows.append(r); print(f"{m['clip']:28s} truth={m['truth']} pred={r['pred']} {r['status']}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = ["clip", "scenario", "subject", "truth", "pred", "status", "quality", "agreement", "should_refuse"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    s = summarize(rows); plot(rows, out); (out / "report.md").write_text(report(s, a)); print(f"\nwrote {out}/report.md")

def plot(rows, out):
    ok = [r for r in rows if r["status"] == "OK" and r["truth"] is not None and not r["should_refuse"]]
    if len(ok) < 2: return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = [(r["pred"] + r["truth"]) / 2 for r in ok]; d = [r["pred"] - r["truth"] for r in ok]
    b, sd = np.mean(d), np.std(d, ddof=1); fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, d); [ax.axhline(v, ls=ls, c=c) for v, ls, c in ((b, "-", "k"), (b + 1.96 * sd, "--", "r"), (b - 1.96 * sd, "--", "r"))]
    ax.set(xlabel="Mean of method and reference (breaths/min)", ylabel="Method minus reference", title="Bland-Altman"); fig.tight_layout()
    fig.savefig(out / "bland_altman.png", dpi=150); plt.close(fig)

def pct(k, n, ci): return f"{k}/{n} ({100*k/n:.0f}%, 95% CI {100*ci[0]:.0f}-{100*ci[1]:.0f}%)" if n else "n/a"
def report(s, a):
    k, n, ci = s["coverage"]; L = [f"# Validation report", "",
        f"Tracks: **{a.tracks}**{' (fixed central ROI, NO SAM: ablation only)' if a.tracks == 'box' else ''}. Split: **{a.split}**. Clips: {s['n_clips']}.",
        f"Subjects: {s['subjects']}. " + ("**Includes non-infant proxies: results do not establish accuracy in infants or sick children.**" if set(s['subjects']) - {'child'} else ""),
        "", "## Accuracy (clips with a reading)", "", f"- Reading returned on {pct(k, n, ci)} of measurable clips; the rest returned UNRELIABLE/no track and would fall back to a retake or manual count."]
    if "mae" in s:
        L += [f"- Mean absolute error: **{s['mae']:.2f} breaths/min** (bootstrap 95% CI {s['mae_ci'][0]:.2f}-{s['mae_ci'][1]:.2f}); worst clip {s['max_abs_err']:.1f}.",
              f"- Bland-Altman: bias {s['bias']:+.2f}, limits of agreement {s['loa'][0]:+.1f} to {s['loa'][1]:+.1f} (see bland_altman.png)."]
    L += ["", "## Threshold agreement", ""]
    for t, v in s["thresholds"].items():
        L += [f"- At {t}/min: {pct(v['agree'], v['n'], v['ci'])} on the correct side. **Missed fast breathing: {len(v['missed'])}** {v['missed']}. False alarms: {len(v['false_alarm'])}. Clips within 3 of the line: {v['near_threshold']}."]
    st_ = s["stress"]; L += ["", "## Stress clips (should be refused)", "",
        f"- Correctly refused: {pct(st_['refused'], st_['n'], st_['ci'])}. **Accepted when it should refuse:** {st_['false_accepts']}", "", "## By scenario", ""]
    L += [f"- {k}: {v['ok']}/{v['n']} gave a reading" for k, v in s["by_scenario"].items()]
    if s["rater_mad"] is not None: L += ["", f"Human counters disagreed by {s['rater_mad']:.1f} breaths/min on average: the floor for any error claim."]
    L += ["", "## Limits", "", "- The quality gate (Spark) is not evaluated here. Test it separately on real frames.",
          "- Thresholds in `estimate.py` must be tuned on `dev` clips only; report `test` only.",
          "- Do not compare with published device sensitivity unless the test design matches."]
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--clips", required=True)
    p.add_argument("--tracks", choices=["api", "cache", "box"], default="cache"); p.add_argument("--cache", default="eval/track_cache")
    p.add_argument("--split", default="test"); p.add_argument("--out", default="eval/out"); asyncio.run(run(p.parse_args()))
