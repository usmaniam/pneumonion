"""Transcoded MP4 + SAM tracks -> breath-rate result with overlay data for the UI."""
import cv2, numpy as np
from .roi import plan, masks_for
from .estimate import chest_motion_signal, estimate_rate

def measure(mp4_path, tracks, fs=10.0, flow_width=320):
    cap = cv2.VideoCapture(str(mp4_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(3)); H = int(cap.get(4))
    scale = min(1.0, flow_width / W)
    p = plan(tracks, n, fs)
    def gen():
        for f in range(n):
            ok, frame = cap.read()
            if not ok: return
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale != 1: g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            roi, bg = masks_for(tracks, p, f, W, H, scale)
            yield g, roi, bg
    sig, valid = chest_motion_signal(gen()); cap.release()
    res = estimate_rate(sig, fs, usable=valid, sam_conf=p["mean_conf"])
    res["overlay"] = [{"f": f, "box": list(tracks[f].box), "usable": bool(p["usable"][f])}
                      for f in sorted(tracks) if f < n]
    res["tracked_fraction"] = round(len(tracks) / max(n, 1), 2)
    return res
