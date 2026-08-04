from typing import TypedDict, Optional, List

class Message(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    question: str
    conversation_history: List[Message]
    policy_no: Optional[str]
    has_policy_no: Optional[bool]
    question_type: Optional[str]
    is_relevant: Optional[bool]
    final_answer: Optional[str]
    # ── Debug fields ─────────────────────────────────────
    # These are only populated when called from debug endpoint
    # Normal chat flow ignores these fields
    debug_mode: Optional[bool]
    source_used: Optional[str]           # "wording" / "schedule" / "wording_and_schedule" / "blocked"
    wording_chunks: Optional[List[str]]  # raw chunks retrieved from ChromaDB
    schedule_text: Optional[str]         # full schedule text retrieved
    resolved_question: Optional[str]     # rewritten question after history resolution
    routing_info: Optional[dict]         # classifier decisions