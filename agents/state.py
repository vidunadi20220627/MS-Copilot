from typing import TypedDict, Optional

class AgentState(TypedDict):
    question: str
    policy_no: Optional[str]
    question_type: Optional[str]  # "wording", "schedule", "transaction"
    final_answer: Optional[str]