from agents import supervisor


class FakeGraph:
    def invoke(self, state):
        return {
            "question": state["question"],
            "conversation_history": state["conversation_history"],
            "policy_no": None,
            "has_policy_no": False,
            "question_type": "wording_only",
            "is_relevant": True,
            "final_answer": "Coverage includes medical expenses.",
            "source_used": "wording_only",
            "routing_info": {"route": "wording_only", "reason": "no policy number"},
            "resolved_question": "What does the policy cover?",
            "wording_chunks": [{"chunk_id": 1, "content": "Coverage includes medical expenses."}],
            "schedule_text": None,
        }


def test_run_supervisor_debug_mode_returns_metadata(monkeypatch):
    monkeypatch.setattr(supervisor, "supervisor_graph", FakeGraph())

    result = supervisor.run_supervisor("What does the policy cover?", debug_mode=True)

    assert isinstance(result, dict)
    assert result["final_answer"] == "Coverage includes medical expenses."
    assert result["source_used"] == "wording_only"
    assert result["routing_info"]["reason"] == "no policy number"
    assert result["resolved_question"] == "What does the policy cover?"
