from typing import TypedDict, Optional

class AgentState(TypedDict):
    question: str
    policy_no: Optional[str]        # extracted from question if present
    has_policy_no: Optional[bool]   # True if valid policy no found
    question_type: Optional[str]    # "wording_only" or "wording_and_schedule"
    is_relevant: Optional[bool]     # True if insurance related
    final_answer: Optional[str]