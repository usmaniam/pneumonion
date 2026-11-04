"""SAM 3.1 video tracking via the Responses API, parsed with meta-sam-parser (verified names)."""
from dataclasses import dataclass
import numpy as np
from meta_sam_parser import (parse_responses_stream, video_segmentation_format, decode_mask_to_raster,
                             SegmentationBoxRecord, SegmentationMaskRecord)

@dataclass(frozen=True)
class FrameTrack:
    frame: int
    box: tuple          # left, top, right, bottom (half-open, source pixels)
    raster: np.ndarray  # bool, shape (box h, box w): masks are cropped to the box
    conf: float | None

def noun_for(age_months):
    return "baby" if age_months < 12 else "child"   # one short noun phrase, nothing else

def tracks_from_records(records):
    """Pick the one object seen on most frames; key by frame index, never by line order."""
    boxes, masks = {}, {}
    for r in records:
        key = (r.object_id, r.frame.frame_index)
        if isinstance(r, SegmentationBoxRecord): boxes[key] = r
        elif isinstance(r, SegmentationMaskRecord): masks[key] = r
    counts = {}
    for oid, _ in boxes: counts[oid] = counts.get(oid, 0) + 1
    if not counts:
        return {}
    oid = max(counts, key=counts.get)
    out = {}
    for (o, f), b in boxes.items():
        m = masks.get((o, f))
        if o != oid or m is None: continue
        raw = decode_mask_to_raster(m.mask)
        ras = np.frombuffer(raw, np.uint8).reshape(m.mask.height, m.mask.width).astype(bool)
        out[f] = FrameTrack(f, (b.left, b.top, b.right, b.bottom), ras,
                            m.confidence if m.confidence is not None else b.confidence)
    return out

async def track_video(client, mp4_path, age_months, delete_upload=True):
    """Upload, stream, parse, delete. Zero tracks is a valid result: caller treats it as a gate failure."""
    with open(mp4_path, "rb") as fh:
        up = await client.files.create(file=fh, purpose="user_data")
    try:
        stream = await client.responses.create(
            model="sam-3.1", stream=True,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": noun_for(age_months)},
                {"type": "input_video", "file_id": up.id}]}],
            extra_body={"metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}})
        result = await parse_responses_stream(stream, video_segmentation_format()).final_result()
        return tracks_from_records(result.records), result.diagnostics
    finally:
        if delete_upload:
            try: await client.files.delete(up.id)
            except Exception: pass   # never mask the real error; sweep job retries later
