from typing import TypedDict, Optional, List

class Message(TypedDict):
    role: str     # "user" or "assistant"
    content: str  # message text

class AgentState(TypedDict):
    question: str
    conversation_history: List[Message]  # full chat history
    policy_no: Optional[str]             # extracted from question OR history
    has_policy_no: Optional[bool]
    question_type: Optional[str]
    is_relevant: Optional[bool]
    final_answer: Optional[str]