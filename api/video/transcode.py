"""ffmpeg wrapper: any browser clip -> 10 fps H.264 MP4, <=640 px wide, no audio, faststart."""
import json, subprocess

class VideoError(ValueError):
    pass

def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=30)
    try:
        j = json.loads(r.stdout); s = j["streams"][0]
        d = j.get("format", {}).get("duration")
        return {"width": int(s["width"]), "height": int(s["height"]), "duration": float(d) if d else None}
    except (KeyError, IndexError, ValueError):
        raise VideoError("unreadable video")

def transcode(src, dst, max_s=70, timeout=180):
    # ffmpeg autorotates by default, so SAM sees displayed width/height. MediaRecorder WebM often
    # lacks a duration header, so the length cap is enforced on the output instead.
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(max_s + 1),
           "-vf", "scale='min(640,iw)':-2,fps=10", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "26", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise VideoError("transcode timed out")
    if r.returncode:
        raise VideoError("could not decode video")
    info = probe(dst)
    if not info["duration"] or info["duration"] > max_s:
        raise VideoError(f"clip longer than {max_s}s")
    if info["duration"] < 30:
        raise VideoError("clip shorter than 30s")
    return info
