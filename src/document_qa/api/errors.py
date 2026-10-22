from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message

# Mapping of stable error codes to Persian messages
ERROR_MESSAGES = {
    "conversation_not_found": "این گفتگو پیدا نشد.",
    "document_not_found": "این سند در این گفتگو پیدا نشد.",
    "validation_error": "درخواست معتبر نیست.",
    "internal_error": "خطای پیش‌بینی‌نشده رخ داد."
}

async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}}
    )

async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"code": "validation_error", "message": ERROR_MESSAGES["validation_error"]}}
    )

async def global_exception_handler(request: Request, exc: Exception):
    # In a real app, we would log exc here
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "internal_error", "message": ERROR_MESSAGES["internal_error"]}}
    )