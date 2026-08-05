from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from agents.supervisor import run_supervisor
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import logging
import json

logger = logging.getLogger("debug_route")

router = APIRouter()
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Server-side session storage ───────────────────────────────────────
# Simple in-memory dict — stores conversation history per session_id
# Resets when server restarts — fine for testing purposes
session_store: dict = {}

class DebugRequest(BaseModel):
    question: str
    session_id: str  # pass same session_id for multi-turn testing

class DebugResponse(BaseModel):
    session_id: str
    question: str
    final_answer: str
    routing_info: dict
    source_used: str
    resolved_question: Optional[str] = None
    retrieved_data: dict
    evaluation: dict


def evaluate_response(
    question: str,
    final_answer: str,
    wording_chunks: Optional[List[dict]],
    schedule_text: Optional[str]
) -> dict:
    """
    Use GPT to evaluate the final answer against retrieved data.
    Checks faithfulness, hallucination and relevance.
    """

    retrieved_context = ""

    if wording_chunks:
        retrieved_context += "POLICY WORDING CHUNKS RETRIEVED:\n"
        for chunk in wording_chunks:
            retrieved_context += f"[Chunk {chunk['chunk_id']}] {chunk['content']}\n\n"

    if schedule_text:
        retrieved_context += f"POLICY SCHEDULE TEXT RETRIEVED:\n{schedule_text[:3000]}\n"

    if not retrieved_context:
        return {
            "faithfulness_score": 0,
            "hallucination_score": 100,
            "relevance_score": 0,
            "hallucinated_sentences": [],
            "evaluation_reasoning": "No data was retrieved — cannot evaluate",
            "evaluation_error": None
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert evaluator for an insurance AI assistant.

                    Your job is to evaluate the quality of an AI-generated answer
                    by comparing it against the source data that was retrieved.

                    You must return a JSON object with exactly these fields:
                    {
                        \"faithfulness_score\": <integer 0-100>,
                        \"hallucination_score\": <integer 0-100>,
                        \"relevance_score\": <integer 0-100>,
                        \"hallucinated_sentences\": [<list of sentences in the answer that are NOT supported by the retrieved data>],
                        \"evaluation_reasoning\": \"<brief explanation of your scores>\"
                    }

                    Scoring guide:
                    faithfulness_score:
                    - 100 = every single claim in the answer is directly supported by retrieved data
                    - 50  = some claims supported, some not verifiable
                    - 0   = answer contradicts or completely ignores retrieved data

                    hallucination_score:
                    - 0   = no hallucination, everything grounded in retrieved data
                    - 50  = some information added that is not in retrieved data
                    - 100 = answer is completely fabricated, nothing from retrieved data

                    relevance_score:
                    - 100 = retrieved data perfectly answers the question
                    - 50  = retrieved data partially relevant to question
                    - 0   = retrieved data is completely irrelevant to question

                    hallucinated_sentences:
                    - List any specific sentences from the final answer that
                      contain information NOT present in the retrieved data
                    - Empty list if no hallucination detected

                    Be strict and accurate. This is for an insurance system
                    where incorrect information can cause real harm."""
                },
                {
                    "role": "user",
                    "content": f"""
                    USER QUESTION:
                    {question}

                    RETRIEVED DATA:
                    {retrieved_context}

                    AI FINAL ANSWER:
                    {final_answer}

                    Please evaluate the final answer against the retrieved data.
                    """
                }
            ]
        )

        eval_result = json.loads(response.choices[0].message.content)
        eval_result["evaluation_error"] = None
        return eval_result

    except Exception as e:
        logger.exception("[DEBUG] Evaluation error")
        return {
            "faithfulness_score": -1,
            "hallucination_score": -1,
            "relevance_score": -1,
            "hallucinated_sentences": [],
            "evaluation_reasoning": "Evaluation failed due to an error",
            "evaluation_error": str(e)
        }


@router.post("/debug", response_model=DebugResponse)
async def debug_chat(request: DebugRequest):
    """
    Debug testing endpoint — NOT called by frontend UI.
    Only used via Postman for backend testing.
    """
    session_id = request.session_id
    question = request.question

    logger.info("[DEBUG ROUTE] ================================")
    logger.info("[DEBUG ROUTE] Session: %s", session_id)
    logger.info("[DEBUG ROUTE] Question: %s", question)

    if session_id not in session_store:
        session_store[session_id] = []
        logger.info("[DEBUG ROUTE] New session created: %s", session_id)
    else:
        logger.info("[DEBUG ROUTE] Existing session: %s messages", len(session_store[session_id]))

    history = session_store[session_id]

    try:
        result = run_supervisor(
            question=question,
            conversation_history=history,
            debug_mode=True
        )

        final_answer = result.get("final_answer", "")
        source_used = result.get("source_used") or "unknown"
        wording_chunks = result.get("wording_chunks") or []
        schedule_text = result.get("schedule_text")
        resolved_question = result.get("resolved_question")
        routing_info = result.get("routing_info") or {}

        evaluation = evaluate_response(
            question=question,
            final_answer=final_answer,
            wording_chunks=wording_chunks,
            schedule_text=schedule_text
        )

        session_store[session_id].append({"role": "user", "content": question})
        session_store[session_id].append({"role": "assistant", "content": final_answer})

        if len(session_store[session_id]) > 20:
            session_store[session_id] = session_store[session_id][-20:]

        return DebugResponse(
            session_id=session_id,
            question=question,
            final_answer=final_answer,
            routing_info=routing_info,
            source_used=source_used,
            resolved_question=resolved_question,
            retrieved_data={
                "wording_chunks": wording_chunks,
                "schedule_text_preview": schedule_text[:500] if schedule_text else None,
                "schedule_text_length": len(schedule_text) if schedule_text else 0
            },
            evaluation=evaluation
        )

    except Exception as e:
        logger.exception("[DEBUG ROUTE] Error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": session_id}
        )


@router.delete("/debug/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session history."""
    if session_id in session_store:
        del session_store[session_id]
        logger.info("[DEBUG ROUTE] Session cleared: %s", session_id)
        return {"message": f"Session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}


@router.get("/debug/session/{session_id}")
async def get_session(session_id: str):
    """View current session history."""
    if session_id in session_store:
        return {
            "session_id": session_id,
            "message_count": len(session_store[session_id]),
            "history": session_store[session_id]
        }
    return {"session_id": session_id, "message_count": 0, "history": []}
