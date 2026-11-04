import hashlib, hmac, os

def hash_id(raw: str, salt: str | None = None) -> str:
    """Keyed hash, 64 hex chars: valid as a Meta safety_identifier (<=64) and as the stored caregiver hash."""
    salt = salt or os.environ.get("PSEUDO_SALT", "")
    if not salt:
        raise RuntimeError("PSEUDO_SALT is not set")
    return hmac.new(salt.encode(), raw.encode(), hashlib.sha256).hexdigest()
