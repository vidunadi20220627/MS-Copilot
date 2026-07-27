from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.supervisor import run_supervisor
import traceback

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question or request.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = run_supervisor(request.question)
        return ChatResponse(question=request.question, answer=answer)
    except Exception as e:
        traceback.print_exc()   # ← prints full error to terminal
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Assistant is running"}