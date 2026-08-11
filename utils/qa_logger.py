import json
import os
from datetime import datetime, timezone
from threading import Lock

LOG_DIR = "logs"
QA_LOG_FILE = os.path.join(LOG_DIR, "qa_history.jsonl")

_lock = Lock()

def log_qa(
    question: str,
    answer: str,
    *,
    resolved_question: str = None,
    policy_no: str = None,
    source_used: str = None,       # "wording_only" / "wording_and_schedule" / "transaction" / "blocked"
    routing_info: dict = None,
    wording_chunks: list = None,
    schedule_text_present: bool = None,
    generated_sql: str = None,      # only relevant for the Vanna/transaction path
    conversation_history_length: int = 0,
    conversation_id: str = None,    # NEW — groups messages from the same browser session
) -> None:
    """
    Append one Q&A exchange as a single JSON line.
    Kept generic so the same function logs wording, schedule,
    and (later) transaction/Vanna answers to one shared file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,   # NEW
        "question": question,
        "resolved_question": resolved_question,
        "answer": answer,
        "policy_no": policy_no,
        "source_used": source_used,
        "routing_info": routing_info,
        "generated_sql": generated_sql,
        "wording_chunks_retrieved": len(wording_chunks) if wording_chunks else 0,
        "wording_chunk_previews": [
            (c.get("content", "")[:150] if isinstance(c, dict) else str(c)[:150])
            for c in (wording_chunks or [])
        ],
        "schedule_text_present": schedule_text_present,
        "conversation_history_length": conversation_history_length,
        # left blank for a human reviewer to fill in later during accuracy review
        "reviewer_verdict": None,   # "correct" / "incorrect" / "partial"
        "reviewer_notes": None,
    }

    with _lock:
        with open(QA_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")