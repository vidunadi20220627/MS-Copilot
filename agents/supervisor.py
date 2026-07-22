from langgraph.graph import StateGraph, END
from agents.state import AgentState
from tools.pdf_tool import answer_from_pdf
from tools.policy_schedule_tool import answer_from_schedule
from tools.vanna_tool import answer_from_db
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json
import re

client = OpenAI(api_key=OPENAI_API_KEY)

def classify_question(state: AgentState) -> AgentState:
    """Classify question type and extract policy number"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Analyze the insurance question and return JSON:
                {
                    "type": "wording" or "schedule" or "transaction",
                    "policy_no": "policy number if mentioned or null"
                }

                wording    → policy terms, conditions, coverage rules,
                             exclusions, definitions, claim procedures
                schedule   → coverage dates, premium amount, insured persons,
                             benefit limits, GST, area of cover
                transaction → payments, transaction history, payment status,
                              payment method, failed payments, amounts

                Return ONLY the JSON. Nothing else."""
            },
            {
                "role": "user",
                "content": state["question"]
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    state["question_type"] = result.get("type", "transaction")
    state["policy_no"] = result.get("policy_no")
    return state

def route_question(state: AgentState) -> str:
    """Route to correct tool"""
    if state["question_type"] == "wording":
        return "pdf_tool"
    elif state["question_type"] == "schedule":
        return "schedule_tool"
    return "transaction_tool"

def call_pdf_tool(state: AgentState) -> AgentState:
    """Call PDF wording tool"""
    answer = answer_from_pdf(
        question=state["question"],
        policy_no=state["policy_no"]
    )
    state["final_answer"] = answer
    return state

def call_schedule_tool(state: AgentState) -> AgentState:
    """Call policy schedule tool"""
    answer = answer_from_schedule(
        question=state["question"],
        policy_no=state["policy_no"]
    )
    state["final_answer"] = answer
    return state

def call_transaction_tool(state: AgentState) -> AgentState:
    """Call Vanna transaction tool"""
    answer = answer_from_db(state["question"])
    state["final_answer"] = answer
    return state

def build_graph():
    """Build LangGraph supervisor"""
    graph = StateGraph(AgentState)

    graph.add_node("classifier", classify_question)
    graph.add_node("pdf_tool", call_pdf_tool)
    graph.add_node("schedule_tool", call_schedule_tool)
    graph.add_node("transaction_tool", call_transaction_tool)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_question,
        {
            "pdf_tool": "pdf_tool",
            "schedule_tool": "schedule_tool",
            "transaction_tool": "transaction_tool"
        }
    )

    graph.add_edge("pdf_tool", END)
    graph.add_edge("schedule_tool", END)
    graph.add_edge("transaction_tool", END)

    return graph.compile()

supervisor_graph = build_graph()

def run_supervisor(question: str) -> str:
    """Main entry point"""
    result = supervisor_graph.invoke({
        "question": question,
        "policy_no": None,
        "question_type": None,
        "final_answer": None
    })
    return result["final_answer"]