"""Live check of the Meta APIs, run from your own machine (no Fly needed):
  MODEL_API_KEY=... python -m scripts.live_check [--clip clip.mp4] [--wav answer.wav]
Prints what each call returned so you can close the [VERIFY] items. Spends a few cents (SAM about $0.12 per clip)."""
import argparse, asyncio, base64, os, sys, tempfile
from pathlib import Path
import cv2, numpy as np

async def main(a):
    if not os.environ.get("MODEL_API_KEY"): sys.exit("MODEL_API_KEY is not set")
    from api.meta.client import make_client
    from api.meta.spark import quality_gate
    c = make_client()
    _, jpg = cv2.imencode(".jpg", np.full((240, 320, 3), 128, np.uint8))       # blank frame: expect a valid object with child_visible false
    try:
        r = await quality_gate(c, [base64.b64encode(jpg.tobytes()).decode()] * 3, "livecheck")
        print(f"SPARK gate  OK  passed={r.passed} failures={r.failures}")
    except Exception as e: print("SPARK gate  FAIL", type(e).__name__, e)
    if a.clip:
        from api.video.transcode import transcode
        from api.meta.sam import track_video
        mp4 = Path(tempfile.mkdtemp()) / "c.mp4"
        try:
            info = transcode(a.clip, mp4); tracks, diag = await track_video(c, mp4, 24)
            print(f"SAM         {'OK' if tracks else 'NO TRACKS'}  tracked={len(tracks)}/{int(info['duration'] * 10)} frames diagnostics={diag}")
        except Exception as e: print("SAM         FAIL", type(e).__name__, e)
    if a.wav:
        from api.meta.voice import transcribe
        try: print("VOICE       OK ", repr(await transcribe(a.wav, "English")))
        except Exception as e: print("VOICE       FAIL", type(e).__name__, e)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--clip"); p.add_argument("--wav"); asyncio.run(main(p.parse_args()))
