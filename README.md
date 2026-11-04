# Pneumonion

Phone-video breathing-rate check plus WHO-style danger-sign triage for children under 5. **Triage support, not a diagnosis.**
Not clinically validated. Do not deploy for real use before the checklist below is done.

## Architecture
Browser PWA (`web/dist`) -> FastAPI (`api/`) -> ffmpeg -> Spark 1.3 quality gate -> SAM 3.1 child tracking -> chest-ROI optical flow ->
two-estimator breath rate -> deterministic rules (`api/rules/*.yaml`) -> follow-up comparison -> Spark wording (validated, template fallback).
**Models never decide the triage class.** Rules do. Raw video is deleted right after measurement; only derived numbers are stored (14 days).

## Run
    pip install -r requirements.txt pytest && apt install ffmpeg
    python -m pytest -q                      # ~1 min
    export MODEL_API_KEY=... ACCESS_CODE=... PSEUDO_SALT=$(openssl rand -hex 32)
    uvicorn api.main:create_app --factory --port 8080
Deploy: see DEPLOY.md (live-API check first, then Fly.io).
Demo clock: set `DEMO_MODE=1`, then `POST /api/demo/clock {"offset_hours": 73}`.
Camera needs HTTPS (the Fly URL, or a tunnel).

## Validate (never-cut)
    python -m eval.make_synthetic eval/synthetic     # harness smoke test ONLY; never publish these numbers
    python -m eval.run_eval --manifest eval/manifest.csv --clips eval/clips --tracks api    # SAM, ~$0.12/clip, caches tracks
    python -m eval.run_eval ... --tracks box         # ablation: fixed ROI, shows what SAM adds
Manifest columns: `clip, truth_bpm` (or `truth_a, truth_b` from two counters), `scenario, should_refuse, split, subject, age_months`.
Tune thresholds on `dev` clips only; report `test` only. Include stress clips (dim, blanket, moving, occluded) with `should_refuse=yes`.

## Before any real-world claim
1. Clinician signs off `who_imci.yaml`, `who_followup.yaml`, `return_precautions`, chest-indrawing class; add guideline citation.
2. Record 15+ real paced-breathing clips and 5+ stress clips; calibrate `min_rms` (0.03 px, from synthetic video), `min_peak_ratio`, `agree_bpm`.
3. Test the Spark gate on ~10 real frames including bad ones; confirm `text.format` json_schema shape on Meta's docs.
4. Native-speaker review of every supported advice language; translate the UI (English only today).
5. Real-device test: iOS Safari and Chrome Android recording formats, low light, thick-clothing and infant cases.
6. Privacy and legal review; job state is in SQLite on one machine (lost on volume loss; no HA).

## Known gaps
Chest-indrawing illustration, UI translations (voice works in 25 languages but questions display in English), question read-aloud, offline use, per-case rate limits. Voice transcription request/response shapes are unverified against the live API.
