import os
from datetime import datetime, timedelta, timezone
_offset = timedelta(0)
def set_offset_hours(h: float):
    global _offset
    if os.environ.get("DEMO_MODE") != "1":
        raise PermissionError("demo clock disabled")
    _offset = timedelta(hours=h)
def now() -> datetime:
    return datetime.now(timezone.utc) + _offset
