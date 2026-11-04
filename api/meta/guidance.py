"""Caregiver wording from the rules output. Never raises: any failure falls back to a fixed template."""
import json, re
from .client import SPARK_MODEL

URG_RANK = {"watch": 0, "today": 1, "go_now": 2}
BY_CLASS = {"URGENT_REFERRAL": "go_now", "ASSESS_TODAY": "today", "FAST_BREATHING": "today",
            "NO_SIGNS_FOUND": "watch", "NO_CLASSIFICATION": "watch"}
BY_VERDICT = {"URGENT": "go_now", "WORSE": "today", "NOT_IMPROVING": "today"}
SCHEMA = {"type": "object", "properties": {"urgency": {"type": "string", "enum": list(URG_RANK)},
          "message": {"type": "string"}}, "required": ["urgency", "message"], "additionalProperties": False}
DOSE = re.compile(r"\b\d+([.,]\d+)?\s*(mg|mcg|ml|g|tablets?|drops?|tsp|tbsp)\b|amoxicillin|antibiotic|paracetamol|"
                  r"acetaminophen|ibuprofen|syrup", re.I)
DIAG = re.compile(r"\bno pneumonia\b|\b(does not|doesn't) have pneumonia\b|\b(has|have) pneumonia\b|\bis healthy\b|\bdiagnos", re.I)

def expected_urgency(cls, verdict=None):
    u = BY_CLASS[cls["classification"]]
    if verdict and verdict["verdict"] in BY_VERDICT and URG_RANK[BY_VERDICT[verdict["verdict"]]] > URG_RANK[u]:
        u = BY_VERDICT[verdict["verdict"]]
    return u

def template(cls, cfg, verdict=None, next_due=None):
    parts = [cls["action_phrase"]]
    if verdict: parts.append(verdict["text"])
    if cls["repeat_count"]: parts.append("The count was close to the limit. Please repeat the check.")
    parts.append(cfg["return_precautions"])
    if next_due and expected_urgency(cls, verdict) != "go_now":
        parts.append(f"Recheck at {next_due[:16].replace('T', ' ')} UTC, or sooner if your child gets worse.")
    return " ".join(parts)

def validate(out, cls, verdict, language):
    if not isinstance(out, dict) or set(out) != {"urgency", "message"}: return False
    msg = out["message"]
    if not isinstance(msg, str) or not 20 <= len(msg) <= 700 or DOSE.search(msg): return False
    if out["urgency"] != expected_urgency(cls, verdict): return False   # cross-checks the rules result in any language
    if language.lower().startswith("english") and (DIAG.search(msg) or cls["action_phrase"].lower() not in msg.lower()):
        return False
    return True

async def make_guidance(client, cls, cfg, language, safety_id, verdict=None, next_due=None, prev_bpm=None):
    fallback = {"text": template(cls, cfg, verdict, next_due), "source": "template", "language": "English"}
    ctx = {"classification": cls["classification"], "action_phrase": cls["action_phrase"], "breaths_per_min": cls["inputs"]["bpm"],
           "threshold": cls["threshold_bpm"], "repeat_count": cls["repeat_count"], "danger_signs": cls["danger_present"],
           "follow_up_verdict": verdict["text"] if verdict else None, "previous_bpm": prev_bpm,
           "next_recheck_utc": next_due, "return_precautions": cfg["return_precautions"]}
    prompt = (f"Write 3 to 5 short, calm sentences in {language} for a caregiver of a young child, based ONLY on this JSON. "
              "State the action_phrase meaning first, then the return precautions, then the next recheck time if given. "
              "Do not diagnose, do not say the child does or does not have pneumonia, do not mention any medicine or dose. "
              f"Set urgency to go_now, today or watch to match the action. Return only JSON.\n{json.dumps(ctx)}")
    try:
        resp = await client.responses.create(
            model=SPARK_MODEL, store=False, reasoning={"effort": "low"}, safety_identifier=safety_id, max_output_tokens=2000,
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            text={"format": {"type": "json_schema", "name": "guidance", "schema": SCHEMA, "strict": True}})
        out = json.loads(re.sub(r"^```(?:json)?|```$", "", resp.output_text.strip(), flags=re.M).strip())
    except Exception:
        return fallback
    return {"text": out["message"].strip(), "source": "model", "language": language} if validate(out, cls, verdict, language) else fallback
