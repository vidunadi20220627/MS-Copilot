from fastapi import APIRouter
from api_gateway.routes.chat import router as chat_router
from api_gateway.routes.debug import router as debug_router

api_router = APIRouter()

api_router.include_router(
    chat_router,
    prefix="/api/v1",
    tags=["chat"]
)

api_router.include_router(
    debug_router,
    prefix="/api/v1",
    tags=["debug"]
)