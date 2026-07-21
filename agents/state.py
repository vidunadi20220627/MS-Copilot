from typing import TypedDict, Optional

class AgentState(TypedDict):
    question: str
    question_type: Optional[str]   # "wording" or "schedule"
    final_answer: Optional[str]