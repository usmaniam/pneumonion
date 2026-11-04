import subprocess, numpy as np, pytest, cv2
from meta_sam_parser import video_segmentation_format, CompletedOutcome
from api.meta.sam import tracks_from_records, FrameTrack, noun_for
from api.video.transcode import transcode, probe, VideoError
from api.breath.roi import plan
from api.breath.pipeline import measure

DOC = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334;c=0.9|>"
       "<|mask;x=0;y=0;c=0.9;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>\n")

def test_parse_real_protocol_line():
    p = video_segmentation_format().create_parser(); p.push(DOC)
    t = tracks_from_records(p.finish(CompletedOutcome()).result.records)
    assert list(t) == [0] and t[0].raster.shape == (27, 60) and t[0].conf == 0.9

def test_noun():
    assert noun_for(6) == "baby" and noun_for(30) == "child"

def raw_video(path, fps, secs, fn, w=320, h=240):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(fps * secs)): pr.stdin.write(fn(i / fps).tobytes())
    pr.stdin.close(); pr.wait()

def synth(bpm, w=320, h=240, amp=1.5):
    r = np.random.default_rng(3)
    tex = cv2.GaussianBlur(r.integers(0, 255, (h, w), np.uint8), (0, 0), 2.0)
    bg = cv2.GaussianBlur(r.integers(0, 255, (h, w), np.uint8), (0, 0), 3.0)
    def frame(t):
        dy = amp * np.sin(2 * np.pi * bpm / 60 * t)
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        chest = cv2.warpAffine(tex, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out = bg.copy(); out[60:180, 80:240] = chest[60:180, 80:240]
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return frame

def synth_tracks(n, w=160, h=120, conf=0.9):
    ras = np.ones((h, w), bool)   # "child" box over the textured patch
    return {i: FrameTrack(i, (80, 60, 80 + w, 60 + h), ras, conf) for i in range(n)}

def test_transcode_spec(tmp_path):
    src = tmp_path / "in.mp4"; raw_video(src, 15, 35, synth(30, 640, 480), 640, 480)
    out = tmp_path / "o.mp4"; info = transcode(src, out)
    assert info["width"] <= 640 and 34 < info["duration"] < 36
    with pytest.raises(VideoError):
        s2 = tmp_path / "s.mp4"; raw_video(s2, 10, 5, synth(30)); transcode(s2, tmp_path / "x.mp4")

@pytest.mark.parametrize("bpm", [24, 45])
def test_end_to_end_synthetic(tmp_path, bpm):
    src = tmp_path / "s.mp4"; raw_video(src, 10, 60, synth(bpm))
    res = measure(src, synth_tracks(600))
    assert res["status"] == "OK" and abs(res["breaths_per_min"] - bpm) <= 3, res

def test_untracked_video_is_unreliable(tmp_path):
    src = tmp_path / "s.mp4"; raw_video(src, 10, 40, synth(30))
    assert measure(src, {})["status"] == "UNRELIABLE"

def test_plan_drops_low_conf_and_area_jumps():
    t = synth_tracks(10)
    t[3] = FrameTrack(3, t[3].box, t[3].raster, 0.2)
    big = np.ones((120, 400), bool); t[6] = FrameTrack(6, (0, 0, 400, 120), big, 0.9)
    u = plan(t, 10, 10)["usable"]
    assert not u[3] and not u[6] and u[2] and u[8]
