"""Local single-user demo case journal; no real claims or PII."""
import json
import re
import sqlite3
from datetime import datetime, timezone
from threading import RLock
from hashlib import sha256

STATUSES=("Pending","Investigating","Evidence requested","Escalated","Closed")
CHECKLIST=("Policy verified","Documents reviewed","Related claims reviewed","Independent review")
TAGS=("Documentation","Repeated reference","High value","Anomaly","Follow-up")

class CaseStore:
    def __init__(self,path=":memory:"):
        self.lock=RLock()
        self.db=sqlite3.connect(path,check_same_thread=False,timeout=10)
        self.db.row_factory=sqlite3.Row
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL, payload TEXT NOT NULL,
                previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL)""")

    def record(self,claim_id,status="Pending",assignee="",notes="",tags=(),checklist=(),watch=False):
        if not re.fullmatch(r"CG-\d{6}",claim_id) or status not in STATUSES:
            raise ValueError("Invalid claim or status.")
        if len(assignee)>80 or len(notes)>2000 or not set(tags)<=set(TAGS) or not set(checklist)<=set(CHECKLIST) or not isinstance(watch,bool):
            raise ValueError("Invalid case fields.")
        payload=json.dumps(dict(status=status,assignee=assignee,notes=notes,tags=sorted(set(tags)),checklist=sorted(set(checklist)),watch=watch),sort_keys=True,separators=(",",":"))
        timestamp=datetime.now(timezone.utc).isoformat()
        with self.lock,self.db:
            self.db.execute("BEGIN IMMEDIATE")
            last=self.db.execute("SELECT event_hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
            previous=last[0] if last else "GENESIS"
            digest=self.digest(previous,claim_id,timestamp,payload)
            self.db.execute("INSERT INTO events(claim_id,recorded_at,payload,previous_hash,event_hash) VALUES(?,?,?,?,?)",(claim_id,timestamp,payload,previous,digest))
        return digest

    @staticmethod
    def digest(previous,claim_id,timestamp,payload):
        return sha256(json.dumps([previous,claim_id,timestamp,payload],separators=(",",":")).encode()).hexdigest()

    def history(self,claim_id=None):
        with self.lock:
            rows=self.db.execute("SELECT * FROM events WHERE (? IS NULL OR claim_id=?) ORDER BY id",(claim_id,claim_id)).fetchall()
        return [dict(id=r["id"],claim_id=r["claim_id"],recorded_at=r["recorded_at"],**json.loads(r["payload"]),event_hash=r["event_hash"]) for r in rows]

    def latest(self):
        return {e["claim_id"]:e for e in self.history()}

    def verify(self):
        with self.lock:
            rows=self.db.execute("SELECT * FROM events ORDER BY id").fetchall()
        previous="GENESIS"
        for r in rows:
            if r["previous_hash"]!=previous or r["event_hash"]!=self.digest(previous,r["claim_id"],r["recorded_at"],r["payload"]):
                return False
            previous=r["event_hash"]
        return True

    def close(self):
        with self.lock: self.db.close()
