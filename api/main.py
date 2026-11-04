import asyncio, collections, hmac, json, os, shutil, tempfile, time, uuid
from datetime import datetime
from pathlib import Path
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile)
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from . import clock
from .followup.compare import load_fu, make_ics, overdue
from .jobs import Deps, default_deps, run_job, _finish
from .gate import manual_count_to_bpm
from .privacy import hash_id
from .store.models import Store
from .meta.voice import SUPPORTED, VoiceError, supported, to_wav

class ParseBody(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)
    language: str = Field(max_length=30)

class Answers(BaseModel):   # every item explicit: "unclear" must be re-asked by tap, never assumed "no"
    unable_to_drink: bool; vomits_everything: bool; convulsions: bool; sleepy: bool; stridor: bool; blue_lips: bool
    chest_indrawing: bool
    days_ill: int = Field(ge=0, le=60)
    fever: bool | None = None
    spo2: int | None = Field(default=None, ge=50, le=100)

def create_app(db_path=None, deps: Deps | None = None, config: dict | None = None):
    cfg = {"access_code": os.environ.get("ACCESS_CODE", ""), "rate_per_hour": int(os.environ.get("RATE_PER_HOUR", 12)),
           "daily_cap": int(os.environ.get("DAILY_CAP", 200)), "max_upload_mb": int(os.environ.get("MAX_UPLOAD_MB", 25)),
           "demo_mode": os.environ.get("DEMO_MODE") == "1", "voice_per_hour": int(os.environ.get("VOICE_PER_HOUR", 120)), **(config or {})}
    store = Store(db_path or os.environ.get("DB_PATH", "./pneumonion.db"))
    app = FastAPI(title="Pneumonion", docs_url=None, redoc_url=None)
    app.state.deps = deps or default_deps(); hits = collections.defaultdict(collections.deque)
    store.sweep(app.state.deps.now())

    def guard(x_access_code: str = Header(default="")):
        code = cfg["access_code"]   # no code configured = closed, never open
        if not code or not hmac.compare_digest(x_access_code.encode(), code.encode()):
            raise HTTPException(401, "access code required")
    def who(x_case_token: str = Header(min_length=16, max_length=128)):
        return hash_id(x_case_token)

    def limit(request, d):
        ip = request.headers.get("fly-client-ip") or (request.client.host if request.client else "?")
        q, t = hits[ip], time.monotonic()
        while q and t - q[0] > 3600: q.popleft()
        if len(q) >= cfg["rate_per_hour"]: raise HTTPException(429, "too many assessments, try later")
        if not store.bump_daily(d.now().date().isoformat(), cfg["daily_cap"]): raise HTTPException(429, "daily limit reached")
        q.append(t)

    async def launch(request, bg, kind, upload, age, language, answers_json, chash, case_id=None, failed=0):
        d = request.app.state.deps
        try: answers = Answers.model_validate_json(answers_json).model_dump()
        except ValidationError: raise HTTPException(422, "answers incomplete: every danger-sign item needs an explicit yes or no")
        limit(request, d); store.sweep(d.now())
        wd = Path(tempfile.mkdtemp(prefix="pn_")); cap = cfg["max_upload_mb"] * 1024 * 1024; n = 0
        try:
            with open(wd / "in.bin", "wb") as f:
                while chunk := await upload.read(1 << 20):
                    n += len(chunk)
                    if n > cap: raise HTTPException(413, "video too large")
                    f.write(chunk)
        except BaseException:
            shutil.rmtree(wd, ignore_errors=True); raise
        job_id = uuid.uuid4().hex; store.job_create(job_id, chash, d.now())
        bg.add_task(run_job, d, store, job_id, wd, {"kind": kind, "age_months": age, "language": language, "answers": answers,
                    "case_id": case_id, "caregiver_hash": chash, "failed_attempts": failed})
        return {"job_id": job_id}

    @app.post("/api/assess", status_code=202, dependencies=[Depends(guard)])
    async def assess(request: Request, bg: BackgroundTasks, video: UploadFile = File(...), age_months: int = Form(..., ge=0, le=59),
                     language: str = Form("English", max_length=30), answers: str = Form(...),
                     failed_attempts: int = Form(0, ge=0, le=10), chash: str = Depends(who)):
        return await launch(request, bg, "initial", video, age_months, language, answers, chash, None, failed_attempts)

    @app.post("/api/cases/{case_id}/recheck", status_code=202, dependencies=[Depends(guard)])
    async def recheck(case_id: str, request: Request, bg: BackgroundTasks, video: UploadFile = File(...), answers: str = Form(...),
                      failed_attempts: int = Form(0, ge=0, le=10), chash: str = Depends(who)):
        case = store.get_case(case_id, chash)
        if not case: raise HTTPException(404, "case not found")
        return await launch(request, bg, "recheck", video, case["child_age_months"], case["language"], answers, chash, case_id, failed_attempts)

    @app.post("/api/manual", dependencies=[Depends(guard)])
    async def manual(request: Request, answers: str = Form(...), breaths: int = Form(...), age_months: int | None = Form(None, ge=0, le=59),
                     language: str = Form("English", max_length=30), case_id: str | None = Form(None), chash: str = Depends(who)):
        """Caregiver's full-minute count goes through the same rules, comparison and storage as a video check."""
        d = request.app.state.deps
        try: ans = Answers.model_validate_json(answers).model_dump()
        except ValidationError: raise HTTPException(422, "answers incomplete")
        try: bpm = manual_count_to_bpm(breaths)
        except ValueError as e: raise HTTPException(422, str(e))
        kind = "initial"
        if case_id:
            c = store.get_case(case_id, chash)
            if not c: raise HTTPException(404, "case not found")
            age_months, language, kind = c["child_age_months"], c["language"], "recheck"
        if age_months is None: raise HTTPException(422, "age_months required")
        p = {"kind": kind, "age_months": age_months, "language": language, "answers": ans, "case_id": case_id, "caregiver_hash": chash}
        breath = {"breaths_per_min": bpm, "status": "OK", "quality": 1.0, "window_s": [0, 60], "source": "manual"}
        return await _finish(d, store, None, p, breath, d.now())

    vhits = collections.defaultdict(collections.deque)
    def vlimit(request):
        ip = request.headers.get("fly-client-ip") or (request.client.host if request.client else "?")
        q, t = vhits[ip], time.monotonic()
        while q and t - q[0] > 3600: q.popleft()
        if len(q) >= cfg["voice_per_hour"]: raise HTTPException(429, "too many voice requests, please tap your answers")
        q.append(t)

    @app.get("/api/voice/languages", dependencies=[Depends(guard)])
    def voice_languages(): return {"languages": SUPPORTED}

    @app.post("/api/transcribe", dependencies=[Depends(guard)])
    async def transcribe(request: Request, audio: UploadFile = File(...), language: str = Form(..., max_length=30), chash: str = Depends(who)):
        """One short spoken answer -> text. Audio never touches disk beyond this request."""
        lang = supported(language)
        if not lang: raise HTTPException(422, "voice is not available for this language, please tap your answers")
        vlimit(request); d = request.app.state.deps; wd = Path(tempfile.mkdtemp(prefix="pn_a_")); n = 0
        try:
            with open(wd / "in", "wb") as f:
                while chunk := await audio.read(1 << 16):
                    n += len(chunk)
                    if n > 2 * 1024 * 1024: raise HTTPException(413, "audio too large")
                    f.write(chunk)
            try:
                await asyncio.to_thread(to_wav, wd / "in", wd / "a.wav")
                return {"text": await d.transcribe(wd / "a.wav", lang)}
            except VoiceError: raise HTTPException(503, "voice is unavailable, please tap your answers")
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    @app.post("/api/interview/parse", dependencies=[Depends(guard)])
    async def interview_parse(body: ParseBody, request: Request, chash: str = Depends(who)):
        vlimit(request)
        try: return await request.app.state.deps.parse(body.transcript, body.language, chash)
        except VoiceError: raise HTTPException(503, "voice is unavailable, please tap your answers")

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(guard)])
    def job(job_id: str, chash: str = Depends(who)):
        j = store.job_get(job_id, chash)
        if not j: raise HTTPException(404, "job not found")
        return j

    @app.get("/api/cases/{case_id}", dependencies=[Depends(guard)])
    def case_view(case_id: str, request: Request, chash: str = Depends(who)):
        c = store.get_case(case_id, chash)
        if not c: raise HTTPException(404, "case not found")
        d = request.app.state.deps; checks = store.checks(case_id); created = datetime.fromisoformat(c["created_at"])
        nxt, late = overdue(created, sum(k["kind"] == "recheck" for k in checks), d.now(), load_fu())
        return {"case": {"id": c["id"], "age_months": c["child_age_months"], "language": c["language"], "created_at": c["created_at"]},
                "checks": [{"ts": k["ts"], "kind": k["kind"], "bpm": k["bpm"], "status": k["status"], "classification": k["classification"],
                            "rule_id": k["rule_id"], "threshold_bpm": k["result"]["threshold_bpm"], "comparison": k["comparison"]} for k in checks],
                "next_recheck_utc": nxt.isoformat() if nxt else None, "recheck_overdue": late}

    @app.get("/api/cases/{case_id}/ics", dependencies=[Depends(guard)])
    def ics(case_id: str, chash: str = Depends(who)):
        c = store.get_case(case_id, chash)
        if not c: raise HTTPException(404, "case not found")
        return Response(make_ics(c["id"], datetime.fromisoformat(c["created_at"])), media_type="text/calendar",
                        headers={"Content-Disposition": 'attachment; filename="pneumonion-rechecks.ics"'})

    @app.delete("/api/cases/{case_id}", dependencies=[Depends(guard)])
    def delete(case_id: str, chash: str = Depends(who)):
        if not store.delete_case(case_id, chash): raise HTTPException(404, "case not found")
        return {"deleted": True}

    @app.post("/api/demo/clock", dependencies=[Depends(guard)])
    def demo_clock(body: dict):
        if not cfg["demo_mode"]: raise HTTPException(404)
        clock.set_offset_hours(float(body.get("offset_hours", 0))); return {"now": clock.now().isoformat()}

    @app.get("/healthz")
    def healthz():
        try: store.ok()
        except Exception: return JSONResponse({"ok": False}, 503)
        if not shutil.which("ffmpeg"): return JSONResponse({"ok": False, "reason": "ffmpeg"}, 503)
        return {"ok": True}

    web = Path(__file__).parent.parent / "web" / "dist"
    if web.is_dir(): app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
