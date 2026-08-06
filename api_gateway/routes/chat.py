from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from agents.supervisor import run_supervisor
import logging

logger = logging.getLogger("chat_route")

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    question: str
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    Accepts question + full conversation history
    Returns answer
    """
    if not request.question or request.question.strip() == "":
        return JSONResponse(
            status_code=200,
            content={
                "question": request.question,
                "answer": "Please enter a question so I can help you."
            }
        )

    try:
        # Convert history to list of dicts
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]

        logger.info(f"[CHAT ROUTE] Question: {request.question}")
        logger.info(f"[CHAT ROUTE] History length: {len(history)}")

        # Pass both question and history to supervisor
        answer = run_supervisor(
            question=request.question,
            conversation_history=history
        )

        return ChatResponse(
            question=request.question,
            answer=answer
        )

    except Exception:
        logger.exception("[CHAT ROUTE] Error")
        return JSONResponse(
            status_code=200,
            content={
                "question": request.question,
                "answer": "I encountered an issue processing your request. Please try again."
            }
        )

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Assistant is running"}