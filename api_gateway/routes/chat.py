from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from agents.supervisor import run_supervisor
from utils.qa_logger import log_qa
import logging
import time

logger = logging.getLogger("chat_route")

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Message]] = []
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    response_time_seconds: float

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    Accepts question + full conversation history
    Returns answer, and silently logs the exchange server-side
    for later accuracy review — no user action required
    """
    if not request.question or request.question.strip() == "":
        return JSONResponse(
            status_code=200,
            content={
                "question": request.question,
                "answer": "Please enter a question so I can help you.",
                "response_time_seconds": 0
            }
        )

    start_time = time.perf_counter()

    try:
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]

        logger.info(f"[CHAT ROUTE] Question: {request.question}")
        logger.info(f"[CHAT ROUTE] History length: {len(history)}")

        # debug_mode=True so we get resolved_question, routing_info,
        # wording_chunks etc back for logging. The frontend still only
        # ever receives the plain "answer" field below.
        result = run_supervisor(
            question=request.question,
            conversation_history=history,
            debug_mode=True
        )

        answer = result["final_answer"]

        response_time = round(time.perf_counter() - start_time, 2)
        logger.info(f"[CHAT ROUTE] Response time: {response_time}s")

        log_qa(
            question=request.question,
            answer=answer,
            resolved_question=result.get("resolved_question"),
            policy_no=result.get("policy_no"),
            source_used=result.get("source_used"),
            routing_info=result.get("routing_info"),
            wording_chunks=result.get("wording_chunks"),
            schedule_text_present=bool(result.get("schedule_text")),
            conversation_history_length=len(history),
            conversation_id=request.conversation_id,
        )

        return ChatResponse(
            question=request.question,
            answer=answer,
            response_time_seconds=response_time
        )

    except Exception:
        logger.exception("[CHAT ROUTE] Error")
        response_time = round(time.perf_counter() - start_time, 2)
        return JSONResponse(
            status_code=200,
            content={
                "question": request.question,
                "answer": "I encountered an issue processing your request. Please try again.",
                "response_time_seconds": response_time
            }
        )

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Assistant is running"}