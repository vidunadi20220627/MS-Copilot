from langgraph.graph import StateGraph, END
from agents.state import AgentState
from tools.pdf_tool import answer_from_pdf
from tools.policy_schedule_tool import answer_from_schedule
from openai import OpenAI
from typing import Optional
from config.settings import OPENAI_API_KEY
import json
import re

client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=20.0
)

# ── Product code prefixes for policy number validation ──────────────
# TODO: Add all product codes here when received
# Example format: "DTPS", "DPAI", "DTHA" etc
# These are the valid prefixes at the START of a policy number
VALID_PRODUCT_CODES = [
    "DTPS",  # Travel Per Trip
    "DPAI",  # Personal Accident
    # TODO: add more product codes here
    # "DTHA",
    # "DMOT",
    # etc
]

def extract_policy_no(question: str) -> Optional[str]:
    """
    Extract policy number from user question
    Validates using product code prefixes
    Returns policy_no if found and valid, None otherwise
    """
    # Convert to uppercase for matching
    question_upper = question.upper()

    # Build regex pattern from valid product codes
    # Matches pattern like DTPS26043904 or DPAI26402372
    for code in VALID_PRODUCT_CODES:
        pattern = rf'\b({code}[A-Z0-9]{{6,}})\b'
        match = re.search(pattern, question_upper)
        if match:
            return match.group(1)

    return None

def classify_question(state: AgentState) -> AgentState:
    """
    Step 1: Check if question is insurance related
    Step 2: Extract policy number if present
    Step 3: Determine question type based on policy number
    """
    question = state["question"]

    # Extract policy number first
    policy_no = extract_policy_no(question)
    state["policy_no"] = policy_no
    state["has_policy_no"] = policy_no is not None

    # Ask GPT if question is insurance related
def classify_question(state: AgentState) -> AgentState:
    question = state["question"]
    policy_no = extract_policy_no(question)
    state["policy_no"] = policy_no
    state["has_policy_no"] = policy_no is not None

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """You are a classifier for an insurance AI assistant.
                Analyze the question and return ONLY a JSON object:
                {"is_relevant": true or false}
                is_relevant = true ONLY if the question is about:
                → Insurance policy terms, conditions, coverage, exclusions
                → Policy schedule details (dates, premium, benefits, insured persons)
                → Payment transactions, payment history, payment status
                → Claims related questions
                → General insurance product questions
                is_relevant = false if the question is about anything else.
                Return ONLY the JSON. Nothing else."""},
                {"role": "user", "content": question}
            ]
        )
        content = response.choices[0].message.content
        result = json.loads(content) if content else {}
        state["is_relevant"] = result.get("is_relevant", False)
    except Exception as e:
        print(f"Classification failed: {e}")
        state["is_relevant"] = None

    if state["is_relevant"] is None:
        state["question_type"] = "error"
    elif state["is_relevant"]:
        state["question_type"] = "wording_and_schedule" if state["has_policy_no"] else "wording_only"
    else:
        state["question_type"] = None

    return state

def route_question(state: AgentState) -> str:
    """Route to correct handler based on classification"""
    if not state["is_relevant"]:
        return "blocked"
    if state["question_type"] == "wording_and_schedule":
        return "wording_and_schedule"
    if state["question_type"] == "wording_only":
        return "wording_only"
    return "blocked"

def handle_wording_only(state: AgentState) -> AgentState:
    """
    No policy number in question
    Get LATEST policy wording from DB view
    Answer only from wording document
    """
    answer = answer_from_pdf(
        question=state["question"],
        policy_no=None  # None means get latest from DB
    )
    state["final_answer"] = answer
    return state

def handle_wording_and_schedule(state: AgentState) -> AgentState:
    """
    Policy number found in question
    Get BOTH wording and schedule for that specific policy
    Combine both answers into one response
    """
    policy_no = state["policy_no"]
    question = state["question"]

    # Get answer from policy wording
    wording_answer = answer_from_pdf(
        question=question,
        policy_no=policy_no
    )

    # Get answer from policy schedule
    schedule_answer = answer_from_schedule(
        question=question,
        policy_no=policy_no
    )

    # Combine both answers using GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful insurance assistant.
                You have been given information from two sources:
                1. Policy Wording — general terms and conditions
                2. Policy Schedule — specific details for this policy

                Combine both into one clear, concise answer.
                Do not repeat information.
                If both sources say the same thing, mention it once.
                If they provide different information, include both."""
            },
            {
                "role": "user",
                "content": f"""
                User Question: {question}

                Answer from Policy Wording:
                {wording_answer}

                Answer from Policy Schedule:
                {schedule_answer}

                Please provide one combined clear answer.
                """
            }
        ]
    )

    state["final_answer"] = response.choices[0].message.content
    return state

def handle_blocked(state: AgentState) -> AgentState:
    """Return friendly message for non relevant questions"""
    state["final_answer"] = (
        "I'm sorry, I can only assist with questions related to your "
        "ERGO insurance policies. This includes policy coverage, "
        "terms and conditions, policy schedule details, and payment "
        "transactions. Please ask a question related to your insurance policy."
    )
    return state

def build_graph():
    """Build LangGraph supervisor"""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classifier", classify_question)
    graph.add_node("wording_only", handle_wording_only)
    graph.add_node("wording_and_schedule", handle_wording_and_schedule)
    graph.add_node("blocked", handle_blocked)

    # Set entry point
    graph.set_entry_point("classifier")

    # Add conditional routing
    graph.add_conditional_edges(
        "classifier",
        route_question,
        {
            "wording_only": "wording_only",
            "wording_and_schedule": "wording_and_schedule",
            "blocked": "blocked"
        }
    )

    # All nodes end the graph
    graph.add_edge("wording_only", END)
    graph.add_edge("wording_and_schedule", END)
    graph.add_edge("blocked", END)

    return graph.compile()

supervisor_graph = build_graph()

def run_supervisor(question: str) -> str:
    """Main entry point"""
    result = supervisor_graph.invoke({
        "question": question,
        "policy_no": None,
        "has_policy_no": None,
        "question_type": None,
        "is_relevant": None,
        "final_answer": None
    })
    return result["final_answer"]