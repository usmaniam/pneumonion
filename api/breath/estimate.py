"""Breath-rate estimation from a 1-D chest motion signal. Two estimators must agree."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, detrend

def chest_motion_signal(frames):
    """frames yields (gray, roi_mask|None, bg_mask|None). Mean vertical Farneback flow in the ROI
    minus background-ring flow (camera shake), accumulated to a displacement-like signal.
    Returns (signal, valid) where valid[i] means frames i-1 and i were both usable."""
    import cv2
    sig, valid, prev = [0.0], [False], None
    for gray, roi, bg in frames:
        if prev is not None:
            pg, proi = prev
            if roi is None or proi is None or (roi & proi).sum() < 30:
                sig.append(0.0); valid.append(False)
            else:
                vy = cv2.calcOpticalFlowFarneback(pg, gray, None, .5, 3, 15, 3, 5, 1.2, 0)[..., 1]
                cam = float(vy[bg].mean()) if bg is not None and bg.sum() > 30 else 0.0
                sig.append(float(vy[roi & proi].mean()) - cam); valid.append(True)
        prev = (gray, roi)
    return np.cumsum(sig), np.array(valid)

def estimate_rate(sig, fs, usable=None, min_window_s=30, band=(0.3, 1.2),
                  agree_bpm=4.0, min_peak_ratio=0.35, sam_conf=1.0, min_rms=0.03):
    n = len(sig)
    usable = np.ones(n, bool) if usable is None else np.asarray(usable, bool)
    best = (0, 0, 0)  # longest contiguous usable run
    i = 0
    while i < n:
        if usable[i]:
            j = i
            while j < n and usable[j]: j += 1
            if j - i > best[0]: best = (j - i, i, j)
            i = j
        else:
            i += 1
    frac = float(usable.mean()) if n else 0.0
    bad = {"breaths_per_min": None, "status": "UNRELIABLE", "quality": 0.0,
           "usable_frame_fraction": frac, "window_s": None, "method_agreement_bpm": None}
    if best[0] / fs < min_window_s:
        return bad
    a, b = best[1], best[2]
    x = detrend(np.asarray(sig[a:b], float))
    sos = butter(3, band, btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    rms = float(x.std())   # band-limited motion in flow-image pixels. [CALIBRATE on real dev clips]
    if rms < min_rms:      # rhythm alone is not enough: no real chest motion means no reading
        return bad | {"window_s": [a / fs, b / fs], "rms_px": round(rms, 4)}
    # Estimator 1: median peak interval
    peaks, _ = find_peaks(x, distance=max(1, int(0.5 * fs)), prominence=0.3 * x.std())
    if len(peaks) < 4:
        return bad | {"window_s": [a / fs, b / fs]}
    peak_bpm = 60.0 / np.median(np.diff(peaks) / fs)
    # Estimator 2: Welch dominant frequency
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(30 * fs)), nfft=8192)
    m = (f >= band[0]) & (f <= band[1])
    f, p = f[m], p[m]
    fpk = f[np.argmax(p)]
    welch_bpm = 60.0 * fpk
    ratio = float(p[np.abs(f - fpk) <= 0.1].sum() / p.sum())
    diff = abs(peak_bpm - welch_bpm)
    quality = float(np.clip(0.5 * min(ratio / 0.7, 1) + 0.3 * frac + 0.2 * sam_conf, 0, 1))
    res = {"breaths_per_min": round((peak_bpm + welch_bpm) / 2, 1), "method_agreement_bpm": round(diff, 2),
           "quality": round(quality, 2), "window_s": [round(a / fs, 1), round(b / fs, 1)],
           "usable_frame_fraction": round(frac, 2), "rms_px": round(rms, 3), "status": "OK"}
    sd = x.std() or 1.0
    res["wave"] = {"t0": round(a / fs, 2), "fs": fs / 2, "y": [round(float(v / sd), 2) for v in x[::2]],
                   "peaks": [round(float(a / fs + q / fs), 2) for q in peaks]}
    if diff > agree_bpm or ratio < min_peak_ratio:
        res |= {"status": "UNRELIABLE", "breaths_per_min": None}
    return res
