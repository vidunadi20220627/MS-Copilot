from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_gateway.router import api_router
from config.settings import APP_HOST, APP_PORT, DEBUG
from db.connection import test_connection
from tools.vanna_tool import setup_vanna

app = FastAPI(
    title="Ergo AI Assistant",
    description="AI Assistant for Insurance System — Phase 1",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    print("Starting Ergo AI Assistant...")

    db_ok = test_connection()
    if not db_ok:
        print("⚠️  WARNING: DB not reachable at startup — transaction queries will fail until DB connects. Wording/schedule tools are unaffected.")

    try:
        setup_vanna()
    except Exception as e:
        print(f"⚠️  WARNING: Vanna setup failed — transaction tool disabled for now: {e}")

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
        reload=DEBUG
    )