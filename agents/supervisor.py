from langgraph.graph import StateGraph, END
from agents.state import AgentState
from tools.pdf_tool import answer_from_pdf
from tools.policy_schedule_tool import answer_from_schedule
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json

client = OpenAI(api_key=OPENAI_API_KEY)

def classify_question(state: AgentState) -> AgentState:
    """Classify whether question is about wording or schedule"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Classify the insurance question into one of these:
                - "wording": questions about policy terms, conditions,
                  coverage details, exclusions, definitions, claim procedures
                - "schedule": questions about specific policy details like
                  coverage dates, premium amount, insured persons,
                  benefit limits, GST, area of cover

                Return ONLY a JSON object like this:
                {"type": "wording"} or {"type": "schedule"}
                Nothing else."""
            },
            {
                "role": "user",
                "content": state["question"]
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    state["question_type"] = result.get("type", "schedule")
    return state

def route_question(state: AgentState) -> str:
    """Route to correct tool based on question type"""
    if state["question_type"] == "wording":
        return "pdf_tool"
    return "schedule_tool"

def call_pdf_tool(state: AgentState) -> AgentState:
    """Call PDF wording tool"""
    answer = answer_from_pdf(state["question"])
    state["final_answer"] = answer
    return state

def call_schedule_tool(state: AgentState) -> AgentState:
    """Call policy schedule tool"""
    answer = answer_from_schedule(state["question"])
    state["final_answer"] = answer
    return state

def build_graph():
    """Build LangGraph supervisor"""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classifier", classify_question)
    graph.add_node("pdf_tool", call_pdf_tool)
    graph.add_node("schedule_tool", call_schedule_tool)

    # Set entry point
    graph.set_entry_point("classifier")

    # Add conditional routing
    graph.add_conditional_edges(
        "classifier",
        route_question,
        {
            "pdf_tool": "pdf_tool",
            "schedule_tool": "schedule_tool"
        }
    )

    # Both tools end the graph
    graph.add_edge("pdf_tool", END)
    graph.add_edge("schedule_tool", END)

    return graph.compile()

# Build graph once on startup
supervisor_graph = build_graph()

def run_supervisor(question: str) -> str:
    """Main entry point to run the supervisor"""
    result = supervisor_graph.invoke({
        "question": question,
        "question_type": None,
        "final_answer": None
    })
    return result["final_answer"]