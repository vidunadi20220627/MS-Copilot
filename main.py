from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_gateway.router import api_router
from config.settings import APP_HOST, APP_PORT
from db.connection import test_connection
from tools.vanna_tool import setup_vanna
import logging

logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("Starting Ergo AI Assistant...")
    test_connection()
    setup_vanna()
    print("Ergo AI Assistant ready")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Ergo AI Assistant",
    description="AI Assistant for Insurance System - Phase 1",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "Ergo AI Assistant is running",
        "version": "0.2.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False
    )