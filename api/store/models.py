"""SQLite store. Derived numbers only: no video, names or raw identifiers."""
import json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, caregiver_hash TEXT NOT NULL, child_age_months INTEGER,
  language TEXT, created_at TEXT, closed_at TEXT);
CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, ts TEXT, kind TEXT,
  bpm REAL, quality REAL, window_s TEXT, status TEXT, danger_json TEXT, classification TEXT, rule_id TEXT,
  rules_version TEXT, comparison_json TEXT, result_json TEXT, video_deleted_at TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, caregiver_hash TEXT, status TEXT, stage TEXT, progress INTEGER,
  result_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS daily(day TEXT PRIMARY KEY, n INTEGER);
"""
iso = lambda d: d.isoformat()
J = lambda o: json.dumps(o, separators=(",", ":"))

class Store:
    def __init__(self, path):
        self.path = str(path)
        with self.conn() as c: c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=10); c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c; c.commit()
        finally:
            c.close()

    def ok(self):
        with self.conn() as c: c.execute("SELECT 1 FROM cases LIMIT 1")
        return True

    def create_case(self, case_id, chash, age, lang, now):
        with self.conn() as c:
            c.execute("INSERT INTO cases VALUES(?,?,?,?,?,NULL)", (case_id, chash, age, lang, iso(now)))

    def get_case(self, case_id, chash):
        with self.conn() as c:
            r = c.execute("SELECT * FROM cases WHERE id=? AND caregiver_hash=?", (case_id, chash)).fetchone()
        return dict(r) if r else None

    def add_check(self, case_id, now, kind, breath, answers, cls, comparison, deleted_at):
        with self.conn() as c:
            c.execute("INSERT INTO checks(case_id,ts,kind,bpm,quality,window_s,status,danger_json,classification,rule_id,"
                      "rules_version,comparison_json,result_json,video_deleted_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (case_id, iso(now), kind, breath.get("breaths_per_min"), breath.get("quality"), J(breath.get("window_s")),
                       breath["status"], J(answers), cls["classification"], cls["rule_id"], cls["rules_version"],
                       J(comparison) if comparison else None, J(cls), iso(deleted_at)))

    def checks(self, case_id):
        with self.conn() as c:
            rows = c.execute("SELECT * FROM checks WHERE case_id=? ORDER BY id", (case_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r); d["result"] = json.loads(d.pop("result_json"))
            cj = d.pop("comparison_json"); d["comparison"] = json.loads(cj) if cj else None
            out.append(d)
        return out

    def delete_case(self, case_id, chash):
        with self.conn() as c:
            if not c.execute("SELECT 1 FROM cases WHERE id=? AND caregiver_hash=?", (case_id, chash)).fetchone():
                return False
            c.execute("DELETE FROM checks WHERE case_id=?", (case_id,)); c.execute("DELETE FROM cases WHERE id=?", (case_id,))
        return True

    def sweep(self, now, days=14):
        cut = iso(now - timedelta(days=days))
        with self.conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM cases WHERE created_at<?", (cut,))]
            for i in ids: c.execute("DELETE FROM checks WHERE case_id=?", (i,))
            c.execute("DELETE FROM cases WHERE created_at<?", (cut,))
            c.execute("DELETE FROM jobs WHERE created_at<?", (iso(now - timedelta(days=1)),))
        return len(ids)

    def job_create(self, job_id, chash, now):
        with self.conn() as c:
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?,NULL,?)", (job_id, chash, "running", "queued", 0, iso(now)))

    def job_update(self, job_id, status, stage, progress, result=None):
        with self.conn() as c:
            c.execute("UPDATE jobs SET status=?,stage=?,progress=?,result_json=COALESCE(?,result_json) WHERE id=?",
                      (status, stage, progress, J(result) if result is not None else None, job_id))

    def job_get(self, job_id, chash):
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=? AND caregiver_hash=?", (job_id, chash)).fetchone()
        if not r: return None
        return {"status": r["status"], "stage": r["stage"], "progress": r["progress"],
                "result": json.loads(r["result_json"]) if r["result_json"] else None}

    def bump_daily(self, day, cap):
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO daily VALUES(?,0)", (day,))
            n = c.execute("SELECT n FROM daily WHERE day=?", (day,)).fetchone()[0]
            if n >= cap: return False
            c.execute("UPDATE daily SET n=n+1 WHERE day=?", (day,))
        return True
