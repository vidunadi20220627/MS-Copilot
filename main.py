from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_gateway.router import api_router
from config.settings import APP_HOST, APP_PORT, DEBUG

app = FastAPI(
    title="Ergo AI Assistant",
    description="AI Assistant for Insurance System — Phase 1",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict after demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "Ergo AI Assistant is running",
        "version": "0.1.0",
        "phase": "Demo — Phase 1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG
    )