from langgraph.graph import StateGraph, END
from agents.state import AgentState
from tools.pdf_tool import answer_from_pdf, answer_from_pdf_with_details
from tools.policy_schedule_tool import answer_from_schedule, answer_from_schedule_with_details
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json
import re
import logging
from typing import Optional, List

# ── Logging Setup ────────────────────────────────────────────────────
import os
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/supervisor.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("supervisor")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Product code prefixes ─────────────────────────────────────────────
VALID_PRODUCT_CODES = [
    "DTPS",
    "DPAI",
    #add more product codes here
]

# ── Followup indicators ───────────────────────────────────────────────
FOLLOWUP_INDICATORS = [
    "that", "it", "this", "those", "these",
    "same", "above", "mentioned", "said",
    "tell me more", "what about", "how about",
    "explain", "elaborate", "more about",
    "what does that mean", "can you explain",
    "and what", "what if", "what else",
    "is that", "does that", "will that",
    "how does that", "why is that",
    "what happens", "then what",
    "go on", "continue", "and also",
    "additionally", "furthermore", "also",
    "another thing", "one more", "as well"
]

# ── Helper Functions ──────────────────────────────────────────────────

def extract_policy_no_from_text(text: str) -> Optional[str]:
    text_upper = text.upper()
    for code in VALID_PRODUCT_CODES:
        pattern = rf'\b({code}[A-Z0-9]{{6,}})\b'
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)
    return None

def extract_policy_no_from_history(
    current_question: str,
    history: List[dict]
) -> Optional[str]:
    policy_no = extract_policy_no_from_text(current_question)
    if policy_no:
        logger.info(f"[POLICY EXTRACT] Found in current question: {policy_no}")
        return policy_no

    logger.info("[POLICY EXTRACT] Not in current question — searching history...")
    for message in reversed(history):
        policy_no = extract_policy_no_from_text(message.get("content", ""))
        if policy_no:
            logger.info(f"[POLICY EXTRACT] Found in history: {policy_no}")
            return policy_no

    logger.info("[POLICY EXTRACT] No policy number found in question or history")
    return None

def build_history_for_gpt(history: List[dict]) -> List[dict]:
    recent_history = history[-10:] if len(history) > 10 else history
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in recent_history
    ]

def is_followup_question(question: str) -> bool:
    question_lower = question.lower().strip()

    word_count = len(question_lower.split())
    if word_count <= 4:
        logger.info(f"[FOLLOWUP CHECK] Short question ({word_count} words) - likely a followup")
        return True

    for indicator in FOLLOWUP_INDICATORS:
        if indicator in question_lower:
            logger.info(f"[FOLLOWUP CHECK] Found followup indicator: '{indicator}'")
            return True

    return False

def has_insurance_context_in_history(history: List[dict]) -> bool:
    if not history:
        return False

    recent = history[-6:]

    insurance_context_keywords = [
        "policy", "insurance", "premium", "coverage",
        "claim", "benefit", "ergo", "schedule",
        "wording", "medical", "travel", "accident",
        "DTPS", "DPAI", "grace period", "exclusion",
        "insured", "cover", "deductible", "trip"
    ]

    for message in recent:
        content = message.get("content", "").lower()
        for keyword in insurance_context_keywords:
            if keyword.lower() in content:
                logger.info(f"[HISTORY CHECK] Insurance context found: '{keyword}' in recent history")
                return True

    logger.info("[HISTORY CHECK] No insurance context in recent history")
    return False

# ── Main Classification ───────────────────────────────────────────────

def classify_question(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("conversation_history", [])

    logger.info("[CLASSIFIER] ========================")
    logger.info(f"[CLASSIFIER] Question: {question}")
    logger.info(f"[CLASSIFIER] History length: {len(history)} messages")

    policy_no = extract_policy_no_from_history(question, history)
    state["policy_no"] = policy_no
    state["has_policy_no"] = policy_no is not None
    logger.info(f"[CLASSIFIER] Policy number: {policy_no}")

    was_followup = False

    if is_followup_question(question):
        logger.info("[CLASSIFIER] Detected followup question — checking history")
        was_followup = True

        if has_insurance_context_in_history(history):
            logger.info("[CLASSIFIER] Insurance context in history — marking as relevant")
            state["is_relevant"] = True

            if state["has_policy_no"]:
                state["question_type"] = "wording_and_schedule"
            else:
                state["question_type"] = "wording_only"

            # Store routing info for debug
            state["routing_info"] = {
                "policy_no_detected": policy_no,
                "question_type": state["question_type"],
                "is_relevant": True,
                "was_followup": was_followup,
                "classification_method": "followup_with_history"
            }

            logger.info(f"[CLASSIFIER] Followup result — is_relevant=True | question_type={state['question_type']}")
            return state

        else:
            logger.info("[CLASSIFIER] No insurance context in history — blocking followup")
            state["is_relevant"] = False
            state["question_type"] = None

            state["routing_info"] = {
                "policy_no_detected": policy_no,
                "question_type": None,
                "is_relevant": False,
                "was_followup": was_followup,
                "classification_method": "followup_blocked_no_history"
            }
            return state

    logger.info("[CLASSIFIER] Not a followup — running GPT classification")

    try:
        messages = [
            {
                "role": "system",
                "content": """You are a classifier for an insurance AI assistant.

                Analyze the latest question in context of the conversation
                history and return ONLY a JSON object:
                {"is_relevant": true or false}

                is_relevant = true if the question is about ANY of these:

                INSURANCE TOPICS:
                → Insurance policy terms, conditions, coverage, exclusions
                → Policy schedule details (dates, premium, benefits,
                  insured persons)
                → Payment transactions, payment history, payment status
                → Claims related questions
                → General insurance product questions
                → Follow up questions about previously discussed
                  insurance topics

                INSURANCE KEYWORDS — question is relevant if it contains
                or relates to any of:
                Accident, Accidental, Act of War, Administrative charges,
                ATM, AIDS, Acquired immune deficiency syndrome,
                Opportunistic infection, Malignant neoplasm,
                Acts of Terrorism, Country of residence,
                Child, Children, Common carrier,
                Civil unrest, Riot, Commotion,
                Depreciation scale, Expedition,
                Extreme sports, Sporting activities,
                Golfing equipment, Hostage,
                Household contents, Hospital,
                Hospital confinement, Insured Person,
                Insolvency, Injury, Jewellery, Kidnap,
                Laptop computer, Loss of limb,
                Loss of hearing, Loss of sight,
                Loss of speech, Dental expenses,
                Major travel event, Manual work,
                Medical expenses, Medical practitioner,
                Mountaineering, Natural disasters,
                Nuclear terrorism, Chemical terrorism,
                Biological terrorism, Overseas, Physician,
                Payment card, Partial disablement, Permanent,
                Pre-existing medical condition, Public place,
                Relative, Selected plan, Serious injury,
                Serious sickness, Sickness, Stolen, Strike,
                Total Disablement, Travel companion,
                Travel agent, Trip, Valuables, War,
                Area of Cover, Individual Cover,
                Adult and Child Cover, Family Cover,
                Multiple Individuals, ERGO Assistance,
                Chronic, Claim, COVID-19, ERGO,
                Policy, Premium, Coverage, Benefit,
                Exclusion, Deductible, Endorsement,
                Grace period, Lapse, Renewal, Reinstatement,
                Sum assured, Sum insured, Repatriation,
                Evacuation, Baggage, Passport, Flight delay,
                Trip cancellation, Trip curtailment

                IMPORTANT CONTEXT RULE:
                If the conversation history is about insurance and the
                current question is clearly continuing that conversation
                even without explicit insurance keywords mark as relevant.
                Example: history about policy coverage,
                user asks "what about children?" — still relevant.

                is_relevant = false ONLY if the question is completely
                unrelated to insurance AND there is no insurance context
                in the conversation history.

                Return ONLY the JSON object. Nothing else."""
            }
        ]

        messages.extend(build_history_for_gpt(history))
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages
        )

        result = json.loads(response.choices[0].message.content)
        state["is_relevant"] = result.get("is_relevant", False)
        logger.info(f"[CLASSIFIER] GPT result: {result}")

    except Exception:
        logger.exception("[CLASSIFIER] GPT error")
        state["is_relevant"] = True

    if state["is_relevant"]:
        if state["has_policy_no"]:
            state["question_type"] = "wording_and_schedule"
        else:
            state["question_type"] = "wording_only"
    else:
        state["question_type"] = None

    state["routing_info"] = {
        "policy_no_detected": policy_no,
        "question_type": state["question_type"],
        "is_relevant": state["is_relevant"],
        "was_followup": was_followup,
        "classification_method": "gpt_classifier"
    }

    logger.info(
        f"[CLASSIFIER] Final — is_relevant={state['is_relevant']} | "
        f"question_type={state['question_type']} | policy_no={state['policy_no']}"
    )

    return state

# ── Routing ───────────────────────────────────────────────────────────

def route_question(state: AgentState) -> str:
    route = "blocked"

    if not state["is_relevant"]:
        route = "blocked"
    elif state["question_type"] == "wording_and_schedule":
        route = "wording_and_schedule"
    elif state["question_type"] == "wording_only":
        route = "wording_only"

    logger.info(f"[ROUTER] Routing to: {route}")
    return route

# ── Handlers ──────────────────────────────────────────────────────────

def handle_wording_only(state: AgentState) -> AgentState:
    logger.info("[WORDING ONLY] Handling wording only question")
    logger.info("[WORDING ONLY] No policy number — fetching latest wording from DB")

    history = state.get("conversation_history", [])
    debug_mode = state.get("debug_mode", False)

    if debug_mode:
        # Debug mode — use detailed version to capture chunks
        raw_answer, chunk_details, resolved_q = answer_from_pdf_with_details(
            question=state["question"],
            policy_no=None,
            conversation_history=history
        )
        state["wording_chunks"] = chunk_details
        state["resolved_question"] = resolved_q
        state["source_used"] = "wording"
    else:
        raw_answer = answer_from_pdf(
            question=state["question"],
            policy_no=None,
            conversation_history=history
        )

    logger.info(f"[WORDING ONLY] Raw answer: {raw_answer[:200]}...")

    try:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful ERGO insurance assistant.
                Answer the user's question naturally and conversationally.
                Take into account the conversation history for context.
                If the user refers to something mentioned earlier
                (like 'that policy' or 'the same coverage')
                use the history to understand what they mean.
                Be clear and concise."""
            }
        ]

        messages.extend(build_history_for_gpt(history))

        messages.append({
            "role": "user",
            "content": f"""
            Question: {state['question']}

            Information retrieved from policy wording:
            {raw_answer}

            Please provide a natural conversational answer
            based on this information and our conversation history.
            """
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        final_answer = response.choices[0].message.content
        logger.info(f"[WORDING ONLY] Final answer: {final_answer[:200]}...")
        state["final_answer"] = final_answer

    except Exception:
        logger.exception("[WORDING ONLY] Error")
        state["final_answer"] = raw_answer

    return state

def _schedule_answer_insufficient(answer: str) -> bool:
    """Check if schedule answer indicates it could not find the information"""
    lowered = answer.lower()
    markers = [
        "does not mention", "not specify", "not specified",
        "does not specify", "could not find", "not stated",
        "not clear from", "does not explicitly", "sorry, i could not",
    ]
    return any(m in lowered for m in markers)

def handle_wording_and_schedule(state: AgentState) -> AgentState:
    """
    SCHEDULE-FIRST approach
    Try schedule alone first, only fall back to wording if insufficient
    """
    policy_no = state["policy_no"]
    question = state["question"]
    history = state.get("conversation_history", [])
    debug_mode = state.get("debug_mode", False)

    logger.info(f"[SCHEDULE FIRST] Policy: {policy_no}")

    if debug_mode:
        schedule_answer, schedule_text = answer_from_schedule_with_details(
            question=question,
            policy_no=policy_no
        )
        state["schedule_text"] = schedule_text
    else:
        schedule_answer = answer_from_schedule(question=question, policy_no=policy_no)

    logger.info(f"[SCHEDULE FIRST] Schedule answer: {schedule_answer[:200]}...")

    if not _schedule_answer_insufficient(schedule_answer):
        # Schedule answered it — fast path
        if debug_mode:
            state["source_used"] = "schedule"
        state["final_answer"] = schedule_answer
        return state

    logger.info("[SCHEDULE FIRST] Schedule insufficient — falling back to wording")

    if debug_mode:
        wording_answer, chunk_details, resolved_q = answer_from_pdf_with_details(
            question=question,
            policy_no=policy_no,
            conversation_history=history
        )
        state["wording_chunks"] = chunk_details
        state["resolved_question"] = resolved_q
        state["source_used"] = "wording_and_schedule"
    else:
        wording_answer = answer_from_pdf(
            question=question,
            policy_no=policy_no,
            conversation_history=history
        )

    logger.info(f"[SCHEDULE FIRST] Wording answer: {wording_answer[:200]}...")

    try:
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful ERGO insurance assistant discussing policy {policy_no}.
                You have two pieces of information about this policy: one from the
                policy schedule (specific figures/dates for this policy) and one from
                the general policy wording (terms/definitions/conditions).

                Merge them into ONE natural, direct, conversational answer.
                Do NOT label or mention which piece came from which source
                (never write "Wording:" or "Schedule:" or similar).
                Do not repeat information.
                Prioritize the schedule's specific figures over general wording
                language when they overlap. Use the conversation history if the
                user is referring to something discussed earlier."""
            }
        ]
        messages.extend(build_history_for_gpt(history))
        messages.append({
            "role": "user",
            "content": f"""
            Question: {question}

            Source 1 (schedule): {schedule_answer}

            Source 2 (wording): {wording_answer}

            Give one combined, natural answer with no source labels.
            """
        })

        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        state["final_answer"] = response.choices[0].message.content

    except Exception:
        logger.exception("[SCHEDULE FIRST] Error combining answers")
        state["final_answer"] = schedule_answer

    return state

def handle_blocked(state: AgentState) -> AgentState:
    logger.info(f"[BLOCKED] Question blocked: {state['question']}")
    state["source_used"] = "blocked"
    state["final_answer"] = (
        "I'm sorry, I can only assist with questions related to your "
        "ERGO insurance policies. This includes policy coverage, "
        "terms and conditions, policy schedule details, and payment "
        "transactions. Please ask a question related to your insurance policy."
    )
    return state

# ── Graph ─────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classifier", classify_question)
    graph.add_node("wording_only", handle_wording_only)
    graph.add_node("wording_and_schedule", handle_wording_and_schedule)
    graph.add_node("blocked", handle_blocked)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_question,
        {
            "wording_only": "wording_only",
            "wording_and_schedule": "wording_and_schedule",
            "blocked": "blocked"
        }
    )

    graph.add_edge("wording_only", END)
    graph.add_edge("wording_and_schedule", END)
    graph.add_edge("blocked", END)

    return graph.compile()

supervisor_graph = build_graph()

# ── Entry Points ──────────────────────────────────────────────────────

def run_supervisor(
    question: str,
    conversation_history: List[dict] = None,
    debug_mode: bool = False
) -> dict:
    """
    Main entry point
    Returns dict with final_answer and optional debug info
    """
    if conversation_history is None:
        conversation_history = []

    logger.info("[SUPERVISOR] ================================")
    logger.info(f"[SUPERVISOR] Question: {question}")
    logger.info(f"[SUPERVISOR] History: {len(conversation_history)} messages")
    logger.info(f"[SUPERVISOR] Debug mode: {debug_mode}")
    logger.info("[SUPERVISOR] ================================")

    result = supervisor_graph.invoke({
        "question": question,
        "conversation_history": conversation_history,
        "policy_no": None,
        "has_policy_no": None,
        "question_type": None,
        "is_relevant": None,
        "final_answer": None,
        "debug_mode": debug_mode,
        "source_used": None,
        "wording_chunks": None,
        "schedule_text": None,
        "resolved_question": None,
        "routing_info": None
    })

    logger.info(f"[SUPERVISOR] Answer: {result['final_answer'][:200]}...")

    return result