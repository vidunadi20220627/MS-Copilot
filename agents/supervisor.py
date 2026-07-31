from langgraph.graph import StateGraph, END
from agents.state import AgentState
from tools.pdf_tool import answer_from_pdf
from tools.policy_schedule_tool import answer_from_schedule
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json
import re
import logging
from typing import Optional, List

# ── Logging Setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/supervisor.log")
    ]
)
logger = logging.getLogger("supervisor")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Product code prefixes for policy number validation ───────────────
# TODO: Add all product codes here when received
VALID_PRODUCT_CODES = [
    "DTPS",   # Travel Per Trip
    "DPAI",   # Personal Accident
    # TODO: add more product codes here
]

# ── Followup indicators ───────────────────────────────────────────────
# Pronouns and vague references that indicate followup questions
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
    """
    Extract policy number from any text string
    Validates using product code prefixes
    """
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
    """
    Try to find policy number from:
    1. Current question first
    2. If not found look through conversation history
    Allows user to mention policy once and not
    repeat it in every subsequent message
    """
    # Check current question first
    policy_no = extract_policy_no_from_text(current_question)
    if policy_no:
        logger.info(f"[POLICY EXTRACT] Found in current question: {policy_no}")
        return policy_no

    # Not in current question — search history
    logger.info("[POLICY EXTRACT] Not in current question — searching history...")
    for message in reversed(history):  # most recent first
        policy_no = extract_policy_no_from_text(message.get("content", ""))
        if policy_no:
            logger.info(f"[POLICY EXTRACT] Found in history: {policy_no}")
            return policy_no

    logger.info("[POLICY EXTRACT] No policy number found in question or history")
    return None

def build_history_for_gpt(history: List[dict]) -> List[dict]:
    """
    Format conversation history for GPT API
    Only send last 10 messages to avoid token limit
    """
    recent_history = history[-10:] if len(history) > 10 else history
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in recent_history
    ]

def is_followup_question(question: str) -> bool:
    """
    Check if question is a short followup using
    pronouns or vague references
    These need history context to classify properly
    """
    question_lower = question.lower().strip()

    # Very short questions are likely followups
    word_count = len(question_lower.split())
    if word_count <= 4:
        logger.info(
            f"[FOLLOWUP CHECK] Short question ({word_count} words) "
            f"— likely a followup"
        )
        return True

    # Check for followup indicator words
    for indicator in FOLLOWUP_INDICATORS:
        if indicator in question_lower:
            logger.info(
                f"[FOLLOWUP CHECK] Found followup indicator: '{indicator}'"
            )
            return True

    return False

def has_insurance_context_in_history(history: List[dict]) -> bool:
    """
    Check if recent conversation history contains
    insurance related context
    If yes — current followup question is relevant
    """
    if not history:
        return False

    # Check last 6 messages (3 exchanges)
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
                logger.info(
                    f"[HISTORY CHECK] Insurance context found: "
                    f"'{keyword}' in recent history"
                )
                return True

    logger.info("[HISTORY CHECK] No insurance context in recent history")
    return False

# ── Main Classification ───────────────────────────────────────────────

def classify_question(state: AgentState) -> AgentState:
    """
    Step 1: Extract policy number from question OR history
    Step 2: Check if question is a followup with pronouns
            If yes and history has insurance context → relevant
    Step 3: If not followup → classify with GPT using keywords
    Step 4: Determine question type
    """
    question = state["question"]
    history = state.get("conversation_history", [])

    logger.info(f"[CLASSIFIER] ========================")
    logger.info(f"[CLASSIFIER] Question: {question}")
    logger.info(f"[CLASSIFIER] History length: {len(history)} messages")

    # ── Extract policy number ─────────────────────────────────────
    policy_no = extract_policy_no_from_history(question, history)
    state["policy_no"] = policy_no
    state["has_policy_no"] = policy_no is not None
    logger.info(f"[CLASSIFIER] Policy number: {policy_no}")

    # ── Step 1: Check if followup question ───────────────────────
    # Handle pronouns like "that", "it", "this"
    # before sending to GPT classifier
    if is_followup_question(question):
        logger.info("[CLASSIFIER] Detected followup question — checking history")

        if has_insurance_context_in_history(history):
            logger.info(
                "[CLASSIFIER] Insurance context in history — "
                "marking as relevant "
            )
            state["is_relevant"] = True

            # Set question type based on policy number
            if state["has_policy_no"]:
                state["question_type"] = "wording_and_schedule"
            else:
                state["question_type"] = "wording_only"

            logger.info(
                f"[CLASSIFIER] Followup result — "
                f"is_relevant=True | "
                f"question_type={state['question_type']}"
            )
            return state

        else:
            logger.info(
                "[CLASSIFIER] No insurance context in history — "
                "blocking followup question"
            )
            state["is_relevant"] = False
            state["question_type"] = None
            return state

    # ── Step 2: Normal GPT classification ────────────────────────
    # Not a followup — classify normally with keywords
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

        # Add conversation history for context
        messages.extend(build_history_for_gpt(history))

        # Add current question
        messages.append({
            "role": "user",
            "content": question
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages
        )

        result = json.loads(response.choices[0].message.content)
        state["is_relevant"] = result.get("is_relevant", False)
        logger.info(f"[CLASSIFIER] GPT result: {result}")

    except Exception as e:
        logger.error(f"[CLASSIFIER] GPT error: {e}")
        # Default to relevant if classification fails
        # Better to attempt answer than wrongly block
        state["is_relevant"] = True

    # ── Set question type ─────────────────────────────────────────
    if state["is_relevant"]:
        if state["has_policy_no"]:
            state["question_type"] = "wording_and_schedule"
        else:
            state["question_type"] = "wording_only"
    else:
        state["question_type"] = None

    logger.info(
        f"[CLASSIFIER] Final — "
        f"is_relevant={state['is_relevant']} | "
        f"question_type={state['question_type']} | "
        f"policy_no={state['policy_no']}"
    )

    return state

# ── Routing ───────────────────────────────────────────────────────────

def route_question(state: AgentState) -> str:
    """Route to correct handler based on classification"""
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
    """
    No policy number in question or history
    Get LATEST policy wording from DB
    Answer in context of conversation history
    """
    logger.info("[WORDING ONLY] Handling wording only question")
    logger.info("[WORDING ONLY] No policy number — fetching latest wording from DB")

    history = state.get("conversation_history", [])

    # Get base answer from PDF tool
    raw_answer = answer_from_pdf(
        question=state["question"],
        policy_no=None,
        conversation_history=history
    )

    logger.info(f"[WORDING ONLY] Raw answer: {raw_answer[:200]}...")

    # Use GPT to make answer conversational with history context
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

        # Add history
        messages.extend(build_history_for_gpt(history))

        # Add current question with raw answer as context
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

    except Exception as e:
        logger.error(f"[WORDING ONLY] Error: {e}")
        state["final_answer"] = raw_answer

    return state

def _schedule_answer_insufficient(answer: str) -> bool:
    """Heuristic: does the schedule answer indicate it couldn't find
    the information, meaning we should also check the wording?"""
    lowered = answer.lower()
    markers = [
        "does not mention", "not specify", "not specified",
        "does not specify", "could not find", "not stated",
        "not clear from", "does not explicitly", "sorry, i could not",
    ]
    return any(m in lowered for m in markers)


def handle_wording_and_schedule(state: AgentState) -> AgentState:
    """
    Policy number found in question or history.
    SCHEDULE-FIRST: try the schedule alone (fast path — no PDF search,
    no embeddings, no extra GPT call). Only fall back to the wording
    tool + a merge call when the schedule genuinely can't answer it.
    """
    policy_no = state["policy_no"]
    question = state["question"]
    history = state.get("conversation_history", [])

    logger.info(f"[SCHEDULE FIRST] Policy: {policy_no}")

    schedule_answer = answer_from_schedule(question=question, policy_no=policy_no)
    logger.info(f"[SCHEDULE FIRST] Schedule answer: {schedule_answer[:200]}...")

    if not _schedule_answer_insufficient(schedule_answer):
        # Schedule answered it directly — fast path, skip the PDF tool entirely
        state["final_answer"] = schedule_answer
        return state

    logger.info("[SCHEDULE FIRST] Schedule insufficient — falling back to wording")
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
                (never write "Wording:" or "Schedule:" or similar — the user
                must never see those words).
                Do not repeat information — if both say the same thing, state it once.
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

    except Exception as e:
        logger.error(f"[SCHEDULE FIRST] Error combining answers: {e}")
        state["final_answer"] = schedule_answer  # fall back to schedule alone

    return state

def handle_blocked(state: AgentState) -> AgentState:
    """Return friendly message for non relevant questions"""
    logger.info(f"[BLOCKED] Question blocked: {state['question']}")
    state["final_answer"] = (
        "I'm sorry, I can only assist with questions related to your "
        "ERGO insurance policies. This includes policy coverage, "
        "terms and conditions, policy schedule details, and payment "
        "transactions. Please ask a question related to your insurance policy."
    )
    return state

# ── Graph ─────────────────────────────────────────────────────────────

def build_graph():
    """Build LangGraph supervisor"""
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

# ── Entry Point ───────────────────────────────────────────────────────

def run_supervisor(question: str, conversation_history: List[dict] = None) -> str:
    """
    Main entry point
    Accepts question and full conversation history
    """
    if conversation_history is None:
        conversation_history = []

    logger.info(f"[SUPERVISOR] ================================")
    logger.info(f"[SUPERVISOR] Question: {question}")
    logger.info(f"[SUPERVISOR] History: {len(conversation_history)} messages")
    logger.info(f"[SUPERVISOR] ================================")

    result = supervisor_graph.invoke({
        "question": question,
        "conversation_history": conversation_history,
        "policy_no": None,
        "has_policy_no": None,
        "question_type": None,
        "is_relevant": None,
        "final_answer": None
    })

    logger.info(f"[SUPERVISOR] Answer: {result['final_answer'][:200]}...")
    return result["final_answer"]