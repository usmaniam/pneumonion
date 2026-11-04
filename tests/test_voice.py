import asyncio, json, subprocess, wave, httpx, pytest
from test_api import make, H
from api.meta.voice import extract_text, supported, to_wav, transcribe, VoiceError, SUPPORTED
from api.meta.interview import validate, parse_answers, ITEMS

def test_language_support():
    assert supported("hindi") == "Hindi" and supported("Swahili") is None and len(SUPPORTED) == 25

def test_extract_text_shapes():
    assert extract_text({"text": " hi "}) == "hi" and extract_text("plain") == "plain"
    assert extract_text({"events": [{"type": "turn"}, {"type": "transcript", "text": "a"}, {"type": "transcript", "text": "b"}]}) == "a b"
    with pytest.raises(VoiceError): extract_text({"nope": 1})

def wav(tmp, fmt="webm"):
    p = tmp / f"s.{fmt}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=2", str(p)], check=True); return p

def test_to_wav_is_mono_16bit_24k(tmp_path):
    out = tmp_path / "o.wav"; to_wav(wav(tmp_path), out)
    with wave.open(str(out)) as w: assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
    with pytest.raises(VoiceError): (tmp_path / "bad").write_bytes(b"junk"); to_wav(tmp_path / "bad", tmp_path / "x.wav")

def test_transcribe_request_shape_and_errors(tmp_path):
    w = tmp_path / "a.wav"; to_wav(wav(tmp_path), w); seen = {}
    def h(req):
        seen["auth"] = req.headers["authorization"]; seen["body"] = req.content; return httpx.Response(200, json={"text": "yes he vomits"})
    text = asyncio.run(transcribe(w, "Hindi", "K", httpx.AsyncClient(transport=httpx.MockTransport(h))))
    assert text == "yes he vomits" and seen["auth"] == "Bearer K"
    assert b'name="request"' in seen["body"] and b'name="audio"' in seen["body"] and b"PUSH_TO_TALK" in seen["body"] and b"Hindi" in seen["body"]
    bad = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(VoiceError): asyncio.run(transcribe(w, "English", "K", bad))
    with pytest.raises(VoiceError): asyncio.run(transcribe(w, "Swahili", "K"))

def model_out(**kw):
    a = {k: {"value": "unclear", "evidence": ""} for k in ITEMS}; a.update(kw); return {"answers": a, "days_ill": {"value": "", "evidence": ""}}
T = "He cannot drink anything. No vomiting. He has been coughing for 3 days."

def test_validate_grounded_answers_pass():
    o = model_out(unable_to_drink={"value": "yes", "evidence": "cannot drink anything"}, vomits_everything={"value": "no", "evidence": "No vomiting."})
    o["days_ill"] = {"value": "3", "evidence": "for 3 days"}; r = validate(o, T)
    assert r["answers"]["unable_to_drink"] == "yes" and r["answers"]["vomits_everything"] == "no" and r["days_ill"] == 3

def test_validate_ungrounded_or_silent_becomes_unclear_never_no():
    o = model_out(blue_lips={"value": "no", "evidence": "his lips are pink"}, convulsions={"value": "no", "evidence": ""}, sleepy={"value": "maybe", "evidence": "x"})
    r = validate(o, T)
    assert r["answers"]["blue_lips"] == r["answers"]["convulsions"] == r["answers"]["sleepy"] == "unclear" and "blue_lips" in r["unclear"]
    assert validate({"answers": "garbage"}, T)["unclear"] == list(ITEMS) and validate(None, T)["days_ill"] is None

def test_validate_days_must_be_grounded_and_in_range():
    assert validate({**model_out(), "days_ill": {"value": "3", "evidence": "for 9 days"}}, T)["days_ill"] is None
    assert validate({**model_out(), "days_ill": {"value": "99", "evidence": "for 3 days"}}, T)["days_ill"] is None

class FC:
    def __init__(s, out): s.out, s.calls = out, []
    @property
    def responses(s): return s
    async def create(s, **kw):
        s.calls.append(kw)
        class R: output_text = s.out if isinstance(s.out, str) else json.dumps(s.out)
        return R()

def test_parse_answers_request_shape_and_untrusted_transcript():
    c = FC(model_out(unable_to_drink={"value": "yes", "evidence": "cannot drink"}))
    r = asyncio.run(parse_answers(c, T, "English", "sid")); k = c.calls[0]
    assert r["answers"]["unable_to_drink"] == "yes" and k["store"] is False and "<transcript>" in k["input"][0]["content"][0]["text"]
    with pytest.raises(VoiceError): asyncio.run(parse_answers(FC("not json"), T, "English", "sid"))

def test_endpoints(tmp_path):
    c, _ = make(tmp_path); d = c.app.state.deps
    async def tr(wav_path, lang): return "he cannot drink"
    async def pa(text, lang, sid): return validate(model_out(unable_to_drink={"value": "yes", "evidence": "cannot drink"}), text)
    d.transcribe, d.parse = tr, pa
    a = wav(tmp_path)
    ok = c.post("/api/transcribe", headers=H, data={"language": "English"}, files={"audio": ("a.webm", a.read_bytes(), "audio/webm")})
    assert ok.status_code == 200 and ok.json() == {"text": "he cannot drink"}
    assert c.post("/api/transcribe", headers=H, data={"language": "Swahili"}, files={"audio": ("a", b"x")}).status_code == 422
    assert c.post("/api/transcribe", headers=H, data={"language": "English"}, files={"audio": ("a", b"x" * (2 * 1024 * 1024 + 1))}).status_code == 413
    assert c.post("/api/transcribe", headers={"X-Case-Token": "t" * 32}, data={"language": "English"}, files={"audio": ("a", b"x")}).status_code == 401
    p = c.post("/api/interview/parse", headers=H, json={"transcript": "he cannot drink", "language": "English"}).json()
    assert p["answers"]["unable_to_drink"] == "yes" and "blue_lips" in p["unclear"]
    assert "Hindi" in c.get("/api/voice/languages", headers=H).json()["languages"]

def test_voice_failure_is_503_so_the_app_falls_back_to_taps(tmp_path):
    c, _ = make(tmp_path); d = c.app.state.deps
    async def boom(*a): raise VoiceError("x")
    d.transcribe = d.parse = boom
    assert c.post("/api/transcribe", headers=H, data={"language": "English"}, files={"audio": ("a", wav(tmp_path).read_bytes())}).status_code == 503
    assert c.post("/api/interview/parse", headers=H, json={"transcript": "x", "language": "English"}).status_code == 503
