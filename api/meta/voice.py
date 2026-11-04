"""Muse Voice Transcribe via the file endpoint, server-side key. Never logs audio or transcripts."""
import json, os, subprocess
import httpx

SUPPORTED = ["Arabic", "Bengali", "Dutch", "English", "French", "German", "Hebrew", "Hindi", "Indonesian", "Italian", "Japanese",
             "Kannada", "Korean", "Malay", "Mandarin", "Marathi", "Polish", "Portuguese", "Spanish", "Tagalog", "Tamil", "Telugu",
             "Thai", "Turkish", "Vietnamese"]
URL, MODEL = "https://api.meta.ai/v1/asr/transcribe", "muse-voice-transcribe-1.0"
KEYWORDS = ["pneumonia", "convulsions", "stridor", "breastfeed", "vomit", "fever", "cough", "chest"]

class VoiceError(RuntimeError):
    pass

def supported(lang):
    return next((s for s in SUPPORTED if s.lower() == (lang or "").strip().lower()), None)

def to_wav(src, dst, max_s=30):
    try:
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(max_s), "-vn", "-ac", "1", "-ar", "24000",
                            "-c:a", "pcm_s16le", str(dst)], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise VoiceError("audio conversion timed out")
    if r.returncode: raise VoiceError("could not read audio")

def extract_text(body):
    """Defensive: the response shape is [VERIFY]. Accept plain text, {text}, {transcript} or {events:[{type:transcript,text}]}."""
    if isinstance(body, str): return body.strip()
    if isinstance(body, dict):
        for k in ("text", "transcript"):
            if isinstance(body.get(k), str): return body[k].strip()
        ev = body.get("events")
        if isinstance(ev, list):
            parts = [e["text"] for e in ev if isinstance(e, dict) and e.get("type") == "transcript" and isinstance(e.get("text"), str)]
            if parts: return " ".join(parts).strip()
    raise VoiceError("unrecognised transcription response")

async def transcribe(wav_path, language, api_key=None, http=None):
    lang = supported(language)
    if not lang: raise VoiceError("language not supported")
    req = {"model": MODEL, "mode": "PUSH_TO_TALK", "languageBias": [lang], "keywords": KEYWORDS}
    own = http is None; c = http or httpx.AsyncClient(timeout=30)
    try:
        with open(wav_path, "rb") as fh: audio = fh.read()
        r = await c.post(URL, headers={"Authorization": f"Bearer {api_key or os.environ['MODEL_API_KEY']}"},
                         files={"request": (None, json.dumps(req), "application/json"), "audio": ("a.wav", audio, "audio/wav")})
    except httpx.HTTPError as e:
        raise VoiceError(f"asr request failed: {type(e).__name__}") from None
    finally:
        if own: await c.aclose()
    if r.status_code != 200: raise VoiceError(f"asr http {r.status_code}")
    try: body = r.json()
    except ValueError: body = r.text
    text = extract_text(body)
    if not text: raise VoiceError("empty transcript")
    return text
