"""Retry policy and manual-count fallback. Pure logic, no I/O."""
MAX_RETRIES = 3

def next_step(failed_attempts: int, reason: str) -> dict:
    """failed_attempts counts captures rejected so far, including this one. After 3 retries -> manual count."""
    if failed_attempts > MAX_RETRIES:
        return {"action": "manual_fallback", "retries_left": 0, "reason": reason}
    return {"action": "retry", "retries_left": MAX_RETRIES - failed_attempts + 1, "reason": reason}

def manual_count_to_bpm(breaths: int, seconds: int = 60) -> float:
    """Caregiver counts breaths against the on-screen timer. Only a full-minute count is accepted."""
    if seconds != 60 or not 5 <= breaths <= 120:
        raise ValueError("count breaths for a full 60 seconds (expected 5-120)")
    return float(breaths)
