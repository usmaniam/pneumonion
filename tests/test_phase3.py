import asyncio, cv2, base64, numpy as np, pytest
from api.meta.spark import quality_gate, evaluate_gate, GateUnavailable, SCHEMA
from api.gate import next_step, manual_count_to_bpm
from api.privacy import hash_id
from api.video.frames import sample_frames
from api.rules.engine import classify, DANGER

OK = {"child_visible": True, "chest_exposed": True, "child_calm": True, "camera_steady": True,
      "lighting_ok": True, "retry_message": ""}

class Resp:
    def __init__(self, t): self.output_text = t
class Err(Exception):
    status_code = 400
class FakeClient:
    def __init__(self, outputs): self.outputs, self.calls = list(outputs), []
    @property
    def responses(self): return self
    async def create(self, **kw):
        self.calls.append(kw); o = self.outputs.pop(0)
        if isinstance(o, Exception): raise o
        return Resp(o)
def run(c, **kw): return asyncio.run(quality_gate(c, ["QUJD"] * 5, "id", **kw))

def test_pass_and_request_shape():
    import json
    c = FakeClient([json.dumps(OK)]); r = run(c)
    k = c.calls[0]
    assert r.passed and k["model"] == "muse-spark-1.3" and k["store"] is False
    assert k["reasoning"] == {"effort": "low"} and k["safety_identifier"] == "id"
    assert k["text"]["format"]["schema"] == SCHEMA
    assert sum(p["type"] == "input_image" for p in k["input"][0]["content"]) == 5

def test_any_false_fails_with_message():
    r = evaluate_gate({**OK, "child_calm": False, "retry_message": "Wait until calm."})
    assert not r.passed and r.failures == ["child_calm"] and r.message == "Wait until calm."

def test_string_false_and_missing_fail_closed():
    assert not evaluate_gate({**OK, "lighting_ok": "true"}).passed
    d = dict(OK); del d["chest_exposed"]; assert evaluate_gate(d).failures == ["chest_exposed"]

def test_bad_message_replaced_by_template():
    r = evaluate_gate({**OK, "camera_steady": False, "retry_message": "x" * 500})
    assert "steady" in r.message and len(r.message) < 200

def test_schema_rejected_falls_back_to_json_object():
    import json
    c = FakeClient([Err(), "```json\n" + json.dumps(OK) + "\n```"]); r = run(c)
    assert r.passed and c.calls[1]["text"]["format"]["type"] == "json_object"

@pytest.mark.parametrize("out", ["not json", "[]", Exception("boom")])
def test_unusable_output_is_unavailable_not_pass(out):
    with pytest.raises(GateUnavailable): run(FakeClient([out]))

def test_retry_policy():
    assert [next_step(n, "x")["action"] for n in (1, 2, 3, 4)] == ["retry", "retry", "retry", "manual_fallback"]
    assert next_step(1, "x")["retries_left"] == 3

def test_manual_count_feeds_rules_engine():
    bpm = manual_count_to_bpm(52)
    r = classify(24, bpm, "OK", {k: False for k in DANGER}, False)
    assert r["classification"] == "FAST_BREATHING"
    with pytest.raises(ValueError): manual_count_to_bpm(30, 30)

def test_hash_id_is_keyed_and_64_chars():
    a, b = hash_id("u1", "s1"), hash_id("u1", "s2")
    assert len(a) == 64 and a != b and a == hash_id("u1", "s1")

def test_sample_frames(tmp_path):
    p = tmp_path / "v.mp4"; w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for i in range(50): w.write(np.full((480, 640, 3), i * 4, np.uint8))
    w.release(); fr = sample_frames(p, 5)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(fr[0]), np.uint8), 1)
    assert len(fr) == 5 and max(img.shape[:2]) <= 512
