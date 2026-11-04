import asyncio, json, os, pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
os.environ["PSEUDO_SALT"] = "test-salt"; os.environ["DEMO_MODE"] = "1"
from api import clock
from api.main import create_app
from api.jobs import Deps
from api.meta.spark import GateResult, GateUnavailable
from api.meta.guidance import make_guidance, validate, template
from api.rules.engine import classify, load_cfg, DANGER

H = {"X-Access-Code": "code", "X-Case-Token": "t" * 32}
NO = {k: False for k in DANGER}
ANS = {**NO, "chest_indrawing": False, "days_ill": 2, "fever": True, "spo2": None}

class FakeClient:
    def __init__(self, out="not json"): self.out = out
    @property
    def responses(self): return self
    async def create(self, **kw):
        class R: output_text = self.out
        return R()

class Sim:   # controls what the fake Meta/vision stages return
    bpm, status, gate, tracks = 52.0, "OK", GateResult(True), 600
def make(tmp_path, **cfgo):
    s = Sim()
    d = Deps(transcode=lambda src, dst: (open(dst, "wb").write(b"x"), {"duration": 60.0})[1],
             sample_frames=lambda mp4: ["QUJD"] * 5,
             gate=lambda fr, sid, lang: _gate(s),
             track=lambda mp4, age: _track(s),
             measure=lambda mp4, tr: {"breaths_per_min": s.bpm if s.status == "OK" else None, "status": s.status, "quality": .8,
                                      "window_s": [5, 55], "overlay": []},
             client=lambda: FakeClient(), now=clock.now)
    clock.set_offset_hours(0); tmp_path.mkdir(parents=True, exist_ok=True)
    app = create_app(tmp_path / "t.db", d, {"access_code": "code", "demo_mode": True, **cfgo})
    return TestClient(app), s
async def _gate(s):
    if isinstance(s.gate, Exception): raise s.gate
    return s.gate
async def _track(s): return {i: None for i in range(s.tracks)}, []

def post(c, url="/api/assess", ans=ANS, headers=H, **data):
    form = {"age_months": "24", "answers": json.dumps(ans), **{k: str(v) for k, v in data.items()}}
    return c.post(url, headers=headers, data=form, files={"video": ("a.webm", b"vid", "video/webm")})
def result(c, r, headers=H):
    assert r.status_code == 202, r.text
    j = c.get(f"/api/jobs/{r.json()['job_id']}", headers=headers).json(); assert j["status"] in ("done", "failed"), j
    return j["result"]

def test_requires_access_code(tmp_path):
    c, _ = make(tmp_path)
    assert post(c, headers={"X-Case-Token": "t" * 32}).status_code == 401
    assert c.get("/healthz").status_code == 200
    c2, _ = make(tmp_path / "closed", access_code="")   # no code configured = closed
    assert post(c2).status_code == 401

def test_initial_flow_and_privacy(tmp_path):
    c, s = make(tmp_path); r = result(c, post(c))
    assert r["outcome"] == "done" and r["classification"]["classification"] == "FAST_BREATHING"
    assert r["guidance"]["source"] == "template" and r["action_phrase"] in r["guidance"]["text"] and r["disclaimer"]
    assert len(r["schedule_utc"]) == 3 and r["comparison"] is None
    view = c.get(f"/api/cases/{r['case_id']}", headers=H).json()
    assert view["checks"][0]["bpm"] == 52.0 and view["case"]["age_months"] == 24

def test_missing_answers_rejected(tmp_path):
    c, _ = make(tmp_path); bad = {k: v for k, v in ANS.items() if k != "blue_lips"}
    assert post(c, ans=bad).status_code == 422

def test_gate_failure_then_manual_fallback(tmp_path):
    c, s = make(tmp_path); s.gate = GateResult(False, ["child_calm"], "Wait until calm.")
    r = result(c, post(c, failed_attempts=0)); assert r["outcome"] == "retry" and r["message"] == "Wait until calm."
    r = result(c, post(c, failed_attempts=3)); assert r["outcome"] == "manual_fallback"

def test_gate_unavailable_and_child_not_found(tmp_path):
    c, s = make(tmp_path); s.gate = GateUnavailable("x")
    assert result(c, post(c))["outcome"] == "manual_fallback"
    s.gate = GateResult(True); s.tracks = 100
    assert result(c, post(c))["reason"] == "child_not_found"

def test_unreliable_is_saved_never_no_signs(tmp_path):
    c, s = make(tmp_path); s.status = "UNRELIABLE"; r = result(c, post(c))
    assert r["classification"]["classification"] == "NO_CLASSIFICATION" and r["manual_fallback_available"]

def test_day0_to_day3_recheck_with_clock_jump(tmp_path):
    c, s = make(tmp_path); first = result(c, post(c)); cid = first["case_id"]
    assert c.get(f"/api/cases/{cid}", headers=H).json()["recheck_overdue"] is False
    assert c.post("/api/demo/clock", headers=H, json={"offset_hours": 73}).status_code == 200
    assert c.get(f"/api/cases/{cid}", headers=H).json()["recheck_overdue"] is True
    s.bpm = 47.0
    r = result(c, post(c, f"/api/cases/{cid}/recheck"))
    assert r["comparison"]["verdict"] == "NOT_IMPROVING"
    s.bpm = 30.0; r = result(c, post(c, f"/api/cases/{cid}/recheck"))
    assert r["comparison"]["verdict"] == "IMPROVING" and r["classification"]["classification"] == "NO_SIGNS_FOUND"
    clock.set_offset_hours(0)

def test_new_danger_sign_on_recheck_is_urgent(tmp_path):
    c, s = make(tmp_path); cid = result(c, post(c))["case_id"]
    r = result(c, post(c, f"/api/cases/{cid}/recheck", ans={**ANS, "sleepy": True}))
    assert r["comparison"]["verdict"] == "URGENT" and r["classification"]["classification"] == "URGENT_REFERRAL"

def test_cases_isolated_by_token_ics_and_delete(tmp_path):
    c, s = make(tmp_path); cid = result(c, post(c))["case_id"]
    other = {**H, "X-Case-Token": "u" * 32}
    assert c.get(f"/api/cases/{cid}", headers=other).status_code == 404
    assert c.delete(f"/api/cases/{cid}", headers=other).status_code == 404
    ics = c.get(f"/api/cases/{cid}/ics", headers=H); assert ics.text.count("BEGIN:VEVENT") == 3
    assert c.delete(f"/api/cases/{cid}", headers=H).json() == {"deleted": True}
    assert c.get(f"/api/cases/{cid}", headers=H).status_code == 404

def test_job_not_readable_by_other_caregiver(tmp_path):
    c, _ = make(tmp_path); jid = post(c).json()["job_id"]
    assert c.get(f"/api/jobs/{jid}", headers={**H, "X-Case-Token": "u" * 32}).status_code == 404

def test_upload_cap_rate_limit_and_daily_cap(tmp_path):
    c, _ = make(tmp_path, max_upload_mb=0)
    assert post(c).status_code == 413
    c, _ = make(tmp_path / "a", rate_per_hour=2); assert [post(c).status_code for _ in range(3)] == [202, 202, 429]
    c, _ = make(tmp_path / "b", daily_cap=1); assert [post(c).status_code for _ in range(2)] == [202, 429]

def test_sweep_expires_old_cases(tmp_path):
    c, s = make(tmp_path); cid = result(c, post(c))["case_id"]
    c.post("/api/demo/clock", headers=H, json={"offset_hours": 24 * 15})
    post(c)   # any new assessment sweeps
    assert c.get(f"/api/cases/{cid}", headers=H).status_code == 404
    clock.set_offset_hours(0)

# --- guidance validation ---
CLS = classify(24, 52, "OK", NO, False); CFG = load_cfg()
def g(out, lang="English", cls=CLS):
    return asyncio.run(make_guidance(FakeClient(json.dumps(out)), cls, CFG, lang, "id"))
GOOD = {"urgency": "today", "message": "See a health worker today. Fast breathing can mean pneumonia. Go now if breathing gets harder."}

def test_guidance_valid_model_text_used():
    assert g(GOOD)["source"] == "model"
def test_guidance_rejects_dose_mismatch_and_diagnosis():
    assert g({**GOOD, "message": GOOD["message"] + " Give 5 ml of syrup."})["source"] == "template"
    assert g({**GOOD, "urgency": "watch"})["source"] == "template"
    assert g({**GOOD, "message": "Your child has pneumonia. See a health worker today. Come back soon please."})["source"] == "template"
    assert g({**GOOD, "message": "Short."})["source"] == "template"
def test_guidance_english_requires_action_phrase_but_other_languages_use_urgency_check():
    assert g({**GOOD, "message": "Please visit a clinic today because the breathing is fast. Return if worse."})["source"] == "template"
    assert g({"urgency": "today", "message": "Veuillez voir un agent de santé aujourd'hui. Revenez si l'état s'aggrave."}, "French")["source"] == "model"
def test_guidance_never_downgrades_below_rules():
    urgent = classify(24, 52, "OK", {**NO, "convulsions": True}, False)
    assert g(GOOD, cls=urgent)["source"] == "template"
    assert urgent["action_phrase"] in g(GOOD, cls=urgent)["text"]
def test_template_has_precautions_and_no_diagnosis():
    t = template(CLS, CFG); assert CFG["return_precautions"] in t and "no pneumonia" not in t.lower()

def test_manual_count_path_and_static_app(tmp_path):
    c, s = make(tmp_path)
    r = c.post("/api/manual", headers=H, data={"age_months": "24", "answers": json.dumps(ANS), "breaths": "52"}).json()
    assert r["classification"]["classification"] == "FAST_BREATHING" and r["breath"]["source"] == "manual"
    r2 = c.post("/api/manual", headers=H, data={"answers": json.dumps(ANS), "breaths": "30", "case_id": r["case_id"]}).json()
    assert r2["comparison"] is not None
    assert c.post("/api/manual", headers=H, data={"age_months": "24", "answers": json.dumps(ANS), "breaths": "2"}).status_code == 422
    assert c.get("/").status_code == 200 and "Pneumonion" in c.get("/").text
