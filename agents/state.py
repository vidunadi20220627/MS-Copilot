from typing import TypedDict, Optional, List, Any

class Message(TypedDict):
    role: str     # "user" or "assistant"
    content: str  # message text

class AgentState(TypedDict, total=False):
    question: str
    conversation_history: List[Message]  # full chat history
    policy_no: Optional[str]             # extracted from question OR history
    has_policy_no: Optional[bool]
    question_type: Optional[str]
    is_relevant: Optional[bool]
    final_answer: Optional[str]
    source_used: Optional[str]
    routing_info: Optional[dict]
    resolved_question: Optional[str]
    wording_chunks: Optional[List[dict]]
    schedule_text: Optional[str]
    debug_mode: Optional[bool]