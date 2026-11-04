# Deployment guide (Fly.io)

One machine, one volume, one container. SQLite lives on the volume, so **never scale past one machine**.
This guide was written without access to a Fly account or a Meta key: the Docker build and the live Meta calls have not been run by the author. Steps 1 and 3 are where that gets settled.

## 0. You need
- A Fly.io account with a payment method, and `flyctl` installed (`fly auth login`).
- A Meta Model API key with access to `sam-3.1`, `muse-spark-1.3` (Standard, **not** `-contributor`) and `muse-voice-transcribe-1.0`.
- This repo unzipped, `python -m pytest -q` passing locally.

## 1. Check the Meta APIs from your own machine first (5 minutes, cents)
This settles the unverified assumptions before you deploy anything.

    pip install -r requirements.txt
    export MODEL_API_KEY=...
    python -m scripts.live_check --clip eval/clips/n_30.mp4 --wav answer.wav

Read the output:
- `SPARK gate OK` means structured output works (the code also falls back to plain JSON on a 400).
- `SAM OK tracked=…` means `input_video` with a file id works and tracks come back. `NO TRACKS` means check the clip (child in frame, 30 to 70 s) or the API request shape.
- `VOICE OK '<text>'` means the transcription response was parsed. If it fails with `unrecognised transcription response`, look at the raw JSON and extend `extract_text` in `api/meta/voice.py`.
Fix anything that fails here before step 2.

## 2. First deploy
Pick a region close to your users (the volume is tied to it).

    fly launch --no-deploy --copy-config --name <your-unique-app-name> --region iad
    fly volumes create data --size 1 --region iad
    fly secrets set MODEL_API_KEY=... ACCESS_CODE=<short-code-you-choose> PSEUDO_SALT=$(openssl rand -hex 32)
    fly deploy
    fly scale count 1

Notes:
- `--copy-config` keeps the repo's `fly.toml` (1 GB, 2 shared CPUs, `/data` mount, `/healthz` check). If the name `pneumonion` is free you can omit `--name`, but then edit `app =` in `fly.toml` to match.
- Keep `PSEUDO_SALT` somewhere safe. Changing it later makes every existing case unreachable (hashes no longer match). That is acceptable for a demo, not for continuity.
- The first build downloads Python wheels (OpenCV, SciPy, `meta-sam-parser`) and installs ffmpeg: expect several minutes. If it fails at `pip install`, the error names the package.

## 3. Verify the deployment
    curl -s https://<app>.fly.dev/healthz                               # {"ok":true}
    curl -s -o /dev/null -w "%{http_code}\n" -X POST https://<app>.fly.dev/api/assess   # 401: closed without the code

Then on a phone, open `https://<app>.fly.dev`, enter the access code, and run one full check (use a volunteer, not a patient). Watch `fly logs` during it: you should see no tracebacks, only lines like `job abc12345 failed: <ExceptionType>` if something breaks (the app never logs video, transcripts or keys).
Check: video replay with the chest band, a breath count, a verdict, the `.ics` download, then Delete case.
Camera and microphone only work over HTTPS. Fly provides it (`force_https = true`).

## 4. Before a demo or review
- **Warm machine:** `fly.toml` stops idle machines to save money. For a live demo set `min_machines_running = 1` and `fly deploy`, so the first request is not a cold start.
- **Day-0 to day-3 demo:** `fly secrets set DEMO_MODE=1`, then `POST /api/demo/clock {"offset_hours": 73}` (with the `X-Access-Code` and `X-Case-Token` headers) jumps time forward. Remove it afterwards: `fly secrets unset DEMO_MODE`.
- **Spend limits:** `DAILY_CAP` (default 200 assessments/day) and `RATE_PER_HOUR` (12 per IP) bound the cost. Lower them for a public link: `fly secrets set DAILY_CAP=40`. SAM is the cost driver, about $0.12 per 60-second clip.
- **Memory:** two jobs run at once on 1 GB. If `fly logs` shows the machine killed for memory, `fly secrets set MAX_JOBS=1` (or raise `memory` in `fly.toml` to `2gb`).
- Only share the access code with people you trust. It is the only gate on your Meta bill.

## 5. Operating
| Task | Command |
|---|---|
| Logs | `fly logs` |
| Status / machine health | `fly status`, `fly checks list` |
| Redeploy after a code change | `fly deploy` |
| Change a setting | `fly secrets set KEY=value` (restarts the machine) |
| Shell on the machine | `fly ssh console` |
| Roll back | `fly releases`, then `fly deploy --image <previous image ref>` |
| Volume snapshots | `fly volumes snapshots list` |
| Remove everything | `fly apps destroy <app>` (deletes the volume and all data) |
Data is short-lived by design: cases expire after 14 days and videos are deleted right after measurement, so backups matter little.

## 6. Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `/healthz` returns 503 | `ffmpeg` missing from the image or the database cannot open. `fly ssh console`, run `ffmpeg -version`; check the `/data` mount exists. |
| App says the access code was not accepted | `ACCESS_CODE` secret missing or different. `fly secrets list` shows names only: set it again. |
| "Video recording is not available" | Not on HTTPS, or an in-app browser. Open in Safari or Chrome directly. |
| Upload 413 | Clip larger than `MAX_UPLOAD_MB` (25). Lower the camera bitrate or raise the limit. |
| 429 | Per-IP or daily cap hit. Raise `RATE_PER_HOUR` or `DAILY_CAP`, or wait. |
| Always "The video check is unavailable" | The Spark gate raised `GateUnavailable`: wrong key, model name, or API error. Re-run `scripts.live_check`. |
| Always "We could not find your child" | SAM returned too few tracked frames: child out of frame, too dark, or the SAM request shape is off. Re-run `scripts.live_check --clip`. |
| Voice button missing | Language not in the 25 supported, or no microphone permission. Tapping always works. |
| Cases vanish after a deploy | The volume is not mounted at `/data` or `DB_PATH` is wrong, so SQLite is writing inside the container. |
| `fly deploy` creates two machines | `fly scale count 1`. Two machines mean two separate databases. |

## 7. Before real people use it
This deployment is a demo of a research prototype. Do not point real caregivers at it until the README checklist is done: clinician sign-off on the rules, validation on real clips, a native-speaker review of every advice language, a privacy notice and legal review, and a real-device test on iOS and Android.
Never commit `.env` or paste the Meta key into chat, tickets or screenshots. If the key leaks, rotate it in Meta's console, then `fly secrets set MODEL_API_KEY=...`.
