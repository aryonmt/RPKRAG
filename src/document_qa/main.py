from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from document_qa.core.config import settings
from document_qa.core.logging import setup_logging
from document_qa.api.errors import (
    AppError, 
    app_error_handler, 
    validation_error_handler, 
    global_exception_handler
)
from document_qa.api.routes import conversations, documents, messages
from document_qa.storage.sqlite import init_db
from document_qa.retrieval.qdrant_index import init_qdrant_collection
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.LOG_LEVEL)
    init_db()
    try:
        init_qdrant_collection()
    except Exception as e:
        logger.error(f"Qdrant init failed: {e}")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document QA",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    app.include_router(conversations.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(messages.router, prefix="/api/v1")

    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()