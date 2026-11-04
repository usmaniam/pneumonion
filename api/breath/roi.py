"""Chest ROI from SAM tracks: upper torso of the box, smoothed centroid, quality drops."""
import numpy as np, cv2
from scipy.ndimage import median_filter

HEAD, TORSO_END = 0.25, 0.60   # fractions of box height (spec Section 6)

def plan(tracks, n, fs, min_conf=0.5, max_area_change=0.3):
    usable = np.zeros(n, bool); info = {}; last_area = None
    cen = np.full((n, 2), np.nan)
    for f in sorted(tracks):
        if f >= n: continue
        t = tracks[f]; h = t.raster.shape[0]
        r0, r1 = int(HEAD * h), max(int(TORSO_END * h), int(HEAD * h) + 1)
        sub = t.raster[r0:r1]; area = int(sub.sum())
        if area == 0 or (t.conf is not None and t.conf < min_conf): continue
        if last_area and abs(area - last_area) / last_area > max_area_change: continue
        last_area = area; usable[f] = True
        ys, xs = np.nonzero(sub)
        cen[f] = (xs.mean() + t.box[0], ys.mean() + t.box[1] + r0)
        info[f] = (sub, r0)
    shift = np.zeros((n, 2), int)
    idx = np.nonzero(usable)[0]
    if len(idx) > 2:
        k = max(3, int(fs) | 1)  # ~1 s median, odd
        for c in range(2):
            full = np.interp(np.arange(n), idx, cen[idx, c])
            shift[:, c] = np.round(median_filter(full, size=k, mode="nearest") - full).astype(int)
    confs = [tracks[f].conf for f in idx if tracks[f].conf is not None]
    return {"usable": usable, "info": info, "shift": shift,
            "mean_conf": float(np.mean(confs)) if confs else 1.0}

def masks_for(tracks, p, f, W, H, scale):
    """(roi, bg_ring) at scaled size for frame f; None if unusable."""
    if not p["usable"][f]: return None, None
    t = tracks[f]; sub, r0 = p["info"][f]
    sx, sy = p["shift"][f]
    roi = np.zeros((H, W), np.uint8); body = np.zeros((H, W), np.uint8)
    def put(dst, ras, x0, y0):
        x0, y0 = x0 + sx, y0 + sy
        h, w = ras.shape
        xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + w), min(H, y0 + h)
        if xb > xa and yb > ya: dst[ya:yb, xa:xb] = ras[ya - y0:yb - y0, xa - x0:xb - x0]
    put(roi, sub, t.box[0], t.box[1] + r0); put(body, t.raster, t.box[0], t.box[1])
    if scale != 1:
        sz = (int(W * scale), int(H * scale))
        roi = cv2.resize(roi, sz, interpolation=cv2.INTER_NEAREST); body = cv2.resize(body, sz, interpolation=cv2.INTER_NEAREST)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    ring = (cv2.dilate(body, k(61)) > 0) & ~(cv2.dilate(body, k(21)) > 0)
    return roi > 0, ring
