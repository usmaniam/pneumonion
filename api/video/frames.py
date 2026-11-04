import base64, cv2

def sample_frames(mp4_path, n=5, max_side=512):
    """n JPEG stills spread across the clip (skipping the first/last 5%), as base64 strings."""
    cap = cv2.VideoCapture(str(mp4_path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (0.05 + 0.9 * i / max(n - 1, 1))))
        ok, f = cap.read()
        if not ok: continue
        s = max_side / max(f.shape[:2])
        if s < 1: f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok: out.append(base64.b64encode(jpg.tobytes()).decode())
    cap.release()
    return out
