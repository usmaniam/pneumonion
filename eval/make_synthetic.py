"""SYNTHETIC smoke-test clips to prove the harness runs. Never publish numbers from these."""
import csv, subprocess, sys
from pathlib import Path
import cv2, numpy as np

def video(path, fn, secs=45, fps=10, w=320, h=240):
    pr = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(secs * fps): pr.stdin.write(fn(i / fps, i).tobytes())
    pr.stdin.close(); pr.wait()

def scene(bpm, amp=1.5, shake=0.0, noise=False, w=320, h=240):
    r = np.random.default_rng(7); tex = cv2.GaussianBlur(r.integers(0, 255, (h, w), np.uint8), (0, 0), 2.0)
    bg = cv2.GaussianBlur(r.integers(0, 255, (h, w), np.uint8), (0, 0), 3.0); walk = np.cumsum(r.standard_normal((9999, 2)), 0) * shake
    def f(t, i):
        if noise: return r.integers(0, 255, (h, w, 3), np.uint8)
        M = np.float32([[1, 0, 0], [0, 1, amp * np.sin(2 * np.pi * bpm / 60 * t)]])
        out = bg.copy(); out[60:180, 80:240] = cv2.warpAffine(tex, M, (w, h), borderMode=cv2.BORDER_REFLECT)[60:180, 80:240]
        if shake: out = cv2.warpAffine(out, np.float32([[1, 0, walk[i, 0]], [0, 1, walk[i, 1]]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return f

if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/synthetic"); d.mkdir(parents=True, exist_ok=True); rows = []
    for b in (20, 30, 40, 45, 50, 60): video(d / f"sine_{b}.mp4", scene(b)); rows.append((f"sine_{b}.mp4", b, "normal", "no"))
    video(d / "static.mp4", scene(30, amp=0)); rows.append(("static.mp4", "", "no_motion", "yes"))
    video(d / "noise.mp4", scene(30, noise=True)); rows.append(("noise.mp4", "", "noise", "yes"))
    video(d / "shake.mp4", scene(30, shake=1.5)); rows.append(("shake.mp4", 30, "moving_camera", "no"))
    with open(d / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "truth_bpm", "scenario", "should_refuse", "split", "subject", "age_months"])
        for c, t, s, r in rows: w.writerow([c, t, s, r, "test", "synthetic", 24])
