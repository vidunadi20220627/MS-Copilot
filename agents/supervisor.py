from langgraph.graph import StateGraph, END
from agents.state import AgentState
from tools.pdf_tool import answer_from_pdf
from tools.policy_schedule_tool import answer_from_schedule
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json
import re
import os
import hashlib  # nosec: used for non-security change detection only
import logging
from typing import Optional, List, Any
from vector_store.chroma import get_or_create_collection, delete_collection

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
# Extension point: add product codes here as they're received
VALID_PRODUCT_CODES = [
    "DTPS",   # Travel Per Trip
    "DPAI",   # Personal Accident
    # Extension point: add more product codes here
]

# ── Keywords for Vector Relevance Classification ────────────────────────
# Keywords are loaded from an external file so non-developers can edit them.
# File: config/insurance_keywords.txt  (one keyword per line, # for comments)

_KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "insurance_keywords.txt")

def _load_keywords_from_file() -> List[str]:
    """Load insurance keywords from the external text file.
    Skips blank lines and lines starting with #."""
    filepath = os.path.abspath(_KEYWORDS_FILE)
    if not os.path.exists(filepath):
        logger.error("[KEYWORDS] File not found: %s", filepath)
        return []

    keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                keywords.append(stripped)

    logger.info("[KEYWORDS] Loaded %s keywords from %s", len(keywords), filepath)
    return keywords

def _hash_keywords(keywords: List[str]) -> str:
    """Return a short hash of the keyword list for change detection."""
    content = "\n".join(sorted(keywords))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

INSURANCE_KEYWORDS = _load_keywords_from_file()

# ── Routing reason constants ──────────────────────────────────────────
REASON_POLICY_NUMBER_FOUND = "policy number found"

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
    policy_no = extract_policy_no_from_text(current_question)
    if policy_no:
        logger.info("[POLICY EXTRACT] Found in current question: %s", policy_no)
        return policy_no

    logger.info("[POLICY EXTRACT] Not in current question — searching history...")
    for message in reversed(history):
        policy_no = extract_policy_no_from_text(message.get("content", ""))
        if policy_no:
            logger.info("[POLICY EXTRACT] Found in history: %s", policy_no)
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

    word_count = len(question_lower.split())
    if word_count <= 4:
        logger.info(
            "[FOLLOWUP CHECK] Short question (%s words) — likely a followup",
            word_count
        )
        return True

    for indicator in FOLLOWUP_INDICATORS:
        if indicator in question_lower:
            logger.info(
                "[FOLLOWUP CHECK] Found followup indicator: '%s'",
                indicator
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
                    "[HISTORY CHECK] Insurance context found: '%s' in recent history",
                    keyword
                )
                return True

    logger.info("[HISTORY CHECK] No insurance context in recent history")
    return False

# ── Vector Relevance Classification ───────────────────────────────────

def _init_keyword_collection():
    """Initialize and populate the keyword vector collection.
    Auto-rebuilds if the keywords file has been edited (hash mismatch)."""
    current_hash = _hash_keywords(INSURANCE_KEYWORDS)
    collection = get_or_create_collection(
        "insurance_keywords",
        metadata={"keywords_hash": current_hash}
    )

    # Check if keywords changed since last build
    stored_hash = (collection.metadata or {}).get("keywords_hash")
    needs_rebuild = collection.count() == 0 or stored_hash != current_hash

    if needs_rebuild:
        logger.info("[KEYWORD INIT] Keywords changed or collection empty — rebuilding...")
        delete_collection("insurance_keywords")
        collection = get_or_create_collection(
            "insurance_keywords",
            metadata={"keywords_hash": current_hash}
        )

        # Batch embed all keywords in one API call
        response = client.embeddings.create(
            input=INSURANCE_KEYWORDS,
            model="text-embedding-3-small"
        )
        embeddings = [data.embedding for data in response.data]
        ids = [f"kw_{i}" for i in range(len(INSURANCE_KEYWORDS))]

        collection.add(
            documents=INSURANCE_KEYWORDS,
            embeddings=embeddings,
            ids=ids
        )
        logger.info("[KEYWORD INIT] Added %s keywords to ChromaDB.", len(INSURANCE_KEYWORDS))
    return collection

def _check_keyword_relevance_vector(question: str) -> bool:
    """
    Embed the question and do a vector similarity search against known keywords.
    Returns True if a strong match is found.
    """
    collection = _init_keyword_collection()
    
    # Embed the incoming question
    response = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Query ChromaDB for top 3 closest keywords
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    if results['distances'] and len(results['distances'][0]) > 0:
        best_distance = results['distances'][0][0]
        best_keyword = results['documents'][0][0]
        
        # Threshold: cosine distance < 0.65 means strong semantic similarity
        if best_distance < 0.65:
            logger.info("[VECTOR CLASSIFIER] Match found: '%s' (Distance: %.3f)", best_keyword, best_distance)
            return True
        else:
            logger.info("[VECTOR CLASSIFIER] No strong match. Best was: '%s' (Distance: %.3f)", best_keyword, best_distance)
    
    return False


# ── Classification helpers ──────────────────────────────────────────────

def _set_question_type(state: AgentState) -> None:
    """Determine question_type based on is_relevant and has_policy_no."""
    if not state["is_relevant"]:
        state["question_type"] = None
    elif state["has_policy_no"]:
        state["question_type"] = "wording_and_schedule"
    else:
        state["question_type"] = "wording_only"

def _handle_followup(
    state: AgentState,
    history: List[dict]
) -> AgentState:
    """Handle classification for a detected followup question using history context."""
    logger.info("[CLASSIFIER] Detected followup question — checking history")

    if has_insurance_context_in_history(history):
        logger.info(
            "[CLASSIFIER] Insurance context in history — marking as relevant "
        )
        state["is_relevant"] = True
        _set_question_type(state)

        logger.info(
            "[CLASSIFIER] Followup result — is_relevant=True | question_type=%s",
            state['question_type']
        )
        state["routing_info"] = {
            "route": state['question_type'],
            "reason": "insurance context in history"
        }
        return state

    logger.info(
        "[CLASSIFIER] No insurance context in history — "
        "blocking followup question"
    )
    state["is_relevant"] = False
    state["question_type"] = None
    state["routing_info"] = {"route": "blocked", "reason": "no insurance context in history"}
    return state

def _run_gpt_classification(state: AgentState, question: str, history: List[dict]) -> None:
    """Classify question relevance using GPT when it's not an obvious followup."""
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
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages
        )

        result = json.loads(response.choices[0].message.content)
        state["is_relevant"] = result.get("is_relevant", False)
        logger.info("[CLASSIFIER] GPT result: %s", result)

    except (json.JSONDecodeError, KeyError, IndexError):
        logger.exception("[CLASSIFIER] GPT response parsing error")
        state["is_relevant"] = True
    except Exception:  # noqa: BLE001
        logger.exception("[CLASSIFIER] GPT API error")
        state["is_relevant"] = True

def _routing_reason(state: AgentState) -> str:
    """Build a human-readable reason string for the routing decision."""
    if not state.get("is_relevant"):
        return "question blocked"
    if state.get("has_policy_no"):
        return REASON_POLICY_NUMBER_FOUND
    return "no policy number"

# ── Main Classification ───────────────────────────────────────────────

def classify_question(state: AgentState) -> AgentState:
    """
    Step 1: Extract policy number from question OR history
    Step 2: Check if question is a followup with pronouns
            If yes and history has insurance context → relevant
    Step 3: If not followup → try vector keyword search
    Step 4: If vector search fails, classify with GPT
    Step 5: Determine question type
    """
    question = state["question"]
    history = state.get("conversation_history", [])

    logger.info("[CLASSIFIER] ========================")
    logger.info("[CLASSIFIER] Question: %s", question)
    logger.info("[CLASSIFIER] History length: %s messages", len(history))

    policy_no = extract_policy_no_from_history(question, history)
    state["policy_no"] = policy_no
    state["has_policy_no"] = policy_no is not None
    logger.info("[CLASSIFIER] Policy number: %s", policy_no)

    if is_followup_question(question):
        return _handle_followup(state, history)

    if _check_keyword_relevance_vector(question):
        state["is_relevant"] = True
    else:
        _run_gpt_classification(state, question, history)
        
    _set_question_type(state)

    logger.info(
        "[CLASSIFIER] Final — %s | %s | %s",
        state['is_relevant'],
        state['question_type'],
        state['policy_no']
    )

    state["routing_info"] = {
        "route": route_question(state),
        "reason": _routing_reason(state)
    }

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

    logger.info("[ROUTER] Routing to: %s", route)
    return route

# ── Handlers ──────────────────────────────────────────────────────────

def handle_wording_only(state: AgentState) -> AgentState:
    """
    No policy number in question or history
    Get LATEST policy wording from DB
    Answer directly from policy wording — no extra GPT call needed,
    since answer_from_pdf already returns a clear, direct answer.
    """
    logger.info("[WORDING ONLY] Handling wording only question")
    logger.info("[WORDING ONLY] No policy number — fetching latest wording from DB")

    history = state.get("conversation_history", [])
    debug_mode = state.get("debug_mode", False)

    pdf_result = answer_from_pdf(
        question=state["question"],
        policy_no=None,
        conversation_history=history,
        return_metadata=debug_mode
    )
    if debug_mode and isinstance(pdf_result, dict):
        raw_answer = pdf_result.get("answer", "")
        state["resolved_question"] = pdf_result.get("resolved_question")
        state["wording_chunks"] = pdf_result.get("wording_chunks", [])
    else:
        raw_answer = pdf_result if isinstance(pdf_result, str) else ""
        state["resolved_question"] = None
        state["wording_chunks"] = []

    logger.info("[WORDING ONLY] Final answer: %s...", raw_answer[:200])
    state["final_answer"] = raw_answer

    state["source_used"] = "wording_only"
    state["routing_info"] = state.get("routing_info") or {"route": "wording_only", "reason": "no policy number"}
    state["schedule_text"] = None

    return state

def _schedule_answer_insufficient(answer: str) -> bool:
    """Heuristic: does the schedule answer indicate it couldn't find
    the information, meaning we should also check the wording?"""
    if not answer:
        return True
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
    SCHEDULE-FIRST: most policy-specific questions (dates, premium,
    benefits) can be answered from the schedule alone — no need to
    also search the full wording PDF and run a merge call every time.
    Only fall back to wording (+ merge) when the schedule genuinely
    can't answer it (general terms/definitions/conditions).
    """
    policy_no = state["policy_no"]
    question = state["question"]
    history = state.get("conversation_history", [])
    debug_mode = state.get("debug_mode", False)

    logger.info("[SCHEDULE FIRST] Policy: %s", policy_no)

    schedule_result = answer_from_schedule(
        question=question,
        policy_no=policy_no,
        return_metadata=debug_mode
    )
    if debug_mode and isinstance(schedule_result, dict):
        schedule_answer = schedule_result.get("answer", "")
        state["schedule_text"] = schedule_result.get("schedule_text")
    else:
        schedule_answer = schedule_result if isinstance(schedule_result, str) else ""
        state["schedule_text"] = None

    logger.info("[SCHEDULE FIRST] Schedule answer: %s...", schedule_answer[:200])

    if not _schedule_answer_insufficient(schedule_answer):
        # Schedule answered it directly — fast path, skip wording entirely
        state["final_answer"] = schedule_answer
        state["resolved_question"] = None
        state["wording_chunks"] = []
        state["source_used"] = "schedule"
        state["routing_info"] = state.get("routing_info") or {"route": "wording_and_schedule", "reason": REASON_POLICY_NUMBER_FOUND}
        return state

    logger.info("[SCHEDULE FIRST] Schedule insufficient — falling back to wording")

    wording_result = answer_from_pdf(
        question=question,
        policy_no=policy_no,
        conversation_history=history,
        return_metadata=debug_mode
    )
    if debug_mode and isinstance(wording_result, dict):
        wording_answer = wording_result.get("answer", "")
        state["resolved_question"] = wording_result.get("resolved_question")
        state["wording_chunks"] = wording_result.get("wording_chunks", [])
    else:
        wording_answer = wording_result if isinstance(wording_result, str) else ""
        state["resolved_question"] = None
        state["wording_chunks"] = []

    logger.info("[SCHEDULE FIRST] Wording answer: %s...", wording_answer[:200])

    try:
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful ERGO insurance assistant discussing policy {policy_no}.
                You have two pieces of information about this policy: one from the
                policy schedule (specific figures/dates for this policy) and one from
                the general policy wording (terms/definitions/conditions).

                Merge them into ONE natural, direct answer.
                Do NOT label or mention which piece came from which source.
                Do not repeat information — if both say the same thing, state it once.
                Prioritize the schedule's specific figures over general wording
                language when they overlap."""
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

    except (json.JSONDecodeError, KeyError, IndexError):
        logger.exception("[SCHEDULE FIRST] Error parsing combined answer")
        state["final_answer"] = schedule_answer  # fall back to schedule alone
    except Exception:  # noqa: BLE001
        logger.exception("[SCHEDULE FIRST] Error combining answers")
        state["final_answer"] = schedule_answer  # fall back to schedule alone

    state["source_used"] = "wording_and_schedule"
    state["routing_info"] = state.get("routing_info") or {"route": "wording_and_schedule", "reason": REASON_POLICY_NUMBER_FOUND}
    return state

def handle_blocked(state: AgentState) -> AgentState:
    """Return friendly message for non relevant questions"""
    logger.info("[BLOCKED] Question blocked: %s", state['question'])
    state["final_answer"] = (
        "I'm sorry, I can only assist with questions related to your "
        "ERGO insurance policies. This includes policy coverage, "
        "terms and conditions, policy schedule details, and payment "
        "transactions. Please ask a question related to your insurance policy."
    )
    state["source_used"] = "blocked"
    state["routing_info"] = state.get("routing_info") or {"route": "blocked", "reason": "question blocked"}
    state["resolved_question"] = None
    state["wording_chunks"] = []
    state["schedule_text"] = None
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

def run_supervisor(question: str, conversation_history: List[dict] = None, debug_mode: bool = False) -> Any:
    """
    Main entry point
    Accepts question and full conversation history
    """
    if conversation_history is None:
        conversation_history = []

    logger.info("[SUPERVISOR] ================================")
    logger.info("[SUPERVISOR] Question: %s", question)
    logger.info("[SUPERVISOR] History: %s messages", len(conversation_history))
    logger.info("[SUPERVISOR] ================================")

    result = supervisor_graph.invoke({
        "question": question,
        "conversation_history": conversation_history,
        "policy_no": None,
        "has_policy_no": None,
        "question_type": None,
        "is_relevant": None,
        "final_answer": None,
        "debug_mode": debug_mode
    })

    logger.info("[SUPERVISOR] Answer: %s...", result['final_answer'][:200])
    if debug_mode:
        return result
    return result["final_answer"]