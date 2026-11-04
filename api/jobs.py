"""Assessment pipeline: transcode -> gate -> track -> measure -> classify -> compare -> guidance -> store."""
import asyncio, functools, os, shutil, uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from .rules.engine import classify, load_cfg
from .followup.compare import compare, load_fu, schedule, overdue
from .gate import next_step
from .privacy import hash_id
from .video.transcode import VideoError

DISCLAIMER = "Triage support. Not a diagnosis. If your child is struggling to breathe, go to a health facility now."
RETRY_MSG = {"child_not_found": "We could not find your child in the video. Keep the whole upper body in view and record again.",
             "video_invalid": "We could not use that video. Record about 60 seconds and try again."}

@dataclass
class Deps:
    transcode: Callable; sample_frames: Callable; gate: Callable; track: Callable; measure: Callable
    client: Callable; now: Callable
    sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(int(os.environ.get("MAX_JOBS", 2))))   # 1 GB VM: two jobs at most
    transcribe: Callable | None = None; parse: Callable | None = None

def default_deps():
    from .video.transcode import transcode
    from .video.frames import sample_frames
    from .meta.spark import quality_gate
    from .meta.sam import track_video
    from .meta.client import shared_client
    from .breath.pipeline import measure
    from . import clock
    return Deps(transcode=transcode, sample_frames=sample_frames,
                gate=lambda fr, sid, lang: quality_gate(shared_client(), fr, sid, lang),
                track=lambda mp4, age: track_video(shared_client(), mp4, age),
                measure=measure, client=shared_client, now=clock.now,
                transcribe=lambda wav, lang: __import__("api.meta.voice", fromlist=["x"]).transcribe(wav, lang),
                parse=lambda text, lang, sid: __import__("api.meta.interview", fromlist=["x"]).parse_answers(shared_client(), text, lang, sid))

def _retry(p, reason, message, failures=()):
    step = next_step(p["failed_attempts"] + 1, reason)
    return {"outcome": step["action"], "reason": reason, "message": message, "failures": list(failures), "next": step}

async def run_job(d, store, job_id, workdir, p):
    async with d.sem:
        try:
            res = await _pipeline(d, store, job_id, workdir, p)
            store.job_update(job_id, "done", "done", 100, res)
        except Exception as e:   # log the type only: never video, transcripts or keys
            print(f"job {job_id[:8]} failed: {type(e).__name__}")
            store.job_update(job_id, "failed", "failed", 0, {"outcome": "error", "message": "Something went wrong. Please retry."})
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

async def _pipeline(d, store, job_id, workdir, p):
    stage = lambda s, n: store.job_update(job_id, "running", s, n)
    raw, mp4 = workdir / "in.bin", workdir / "out.mp4"
    chash = p["caregiver_hash"]; safety = chash
    stage("transcoding", 10)
    try:
        info = await asyncio.to_thread(d.transcode, raw, mp4)
    except VideoError as e:
        return _retry(p, "video_invalid", f"{RETRY_MSG['video_invalid']} ({e})")
    raw.unlink(missing_ok=True)
    stage("gating", 25)
    from .meta.spark import GateUnavailable
    frames = await asyncio.to_thread(d.sample_frames, mp4)
    try:
        gate = await d.gate(frames, safety, p["language"])
    except GateUnavailable:
        return {"outcome": "manual_fallback", "reason": "gate_unavailable",
                "message": "The video check is unavailable. Count breaths for one minute with the timer."}
    if not gate.passed:
        return _retry(p, "quality_gate", gate.message, gate.failures)
    stage("tracking", 45)
    tracks, _ = await d.track(mp4, p["age_months"])
    if len(tracks) < 0.5 * info["duration"] * 10:
        return _retry(p, "child_not_found", RETRY_MSG["child_not_found"])
    stage("measuring", 70)
    breath = await asyncio.to_thread(d.measure, mp4, tracks)
    shutil.rmtree(workdir, ignore_errors=True)       # video gone as soon as measurement ends
    return await _finish(d, store, job_id, p, breath, d.now())

async def _finish(d, store, job_id, p, breath, deleted_at):
    chash = p["caregiver_hash"]; safety = chash
    if job_id: store.job_update(job_id, "running", "classifying", 85)
    a = p["answers"]; cfg = load_cfg(); fu = load_fu()
    danger = {k: a[k] for k in ("unable_to_drink", "vomits_everything", "convulsions", "sleepy", "stridor", "blue_lips")}
    cls = classify(p["age_months"], breath["breaths_per_min"], breath["status"], danger, a["chest_indrawing"], a.get("spo2"), cfg)
    now = d.now(); kind = p["kind"]; comparison = prev_bpm = None
    if kind == "initial":
        case_id = uuid.uuid4().hex; store.create_case(case_id, chash, p["age_months"], p["language"], now); created = now
        done_rechecks = 0
    else:
        case = store.get_case(p["case_id"], chash); case_id = case["id"]
        from datetime import datetime
        created = datetime.fromisoformat(case["created_at"]); hist = store.checks(case_id)
        done_rechecks = sum(c["kind"] == "recheck" for c in hist) + 1
        hrs = (now - created).total_seconds() / 3600
        day = int((hrs + fu["day_tolerance_hours"]) // 24)
        comparison = compare(cls, hist[-1]["result"], hist[0]["result"], day, fu)
        prev_bpm = hist[-1]["bpm"]
    nxt, _ = overdue(created, done_rechecks, now, fu)
    nxt_s = nxt.isoformat() if nxt else None
    store.add_check(case_id, now, kind, breath, a, cls, comparison, deleted_at)
    from .meta.guidance import make_guidance
    guidance = await make_guidance(d.client(), cls, cfg, p["language"], safety, comparison, nxt_s, prev_bpm)
    return {"outcome": "done", "case_id": case_id, "classification": cls, "breath": breath, "comparison": comparison,
            "guidance": guidance, "action_phrase": cls["action_phrase"], "next_recheck_utc": nxt_s,
            "schedule_utc": [s.isoformat() for s in schedule(created, fu)],
            "manual_fallback_available": breath["status"] != "OK", "disclaimer": DISCLAIMER}
