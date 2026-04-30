from fastapi import FastAPI
from app.api.routes_chat import router as chat_router

app = FastAPI(
    title="AI Language Tutor",
    description="AI Agent hỗ trợ học ngoại ngữ",
    version="0.1.0"
)

app.include_router(chat_router, prefix="/api", tags=["chat"])

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}