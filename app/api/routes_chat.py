from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.pipeline import run_pipeline

router = APIRouter()

class ChatRequest(BaseModel):
    user_input: str
    user_id: str = "default_user"   # Chuẩn bị cho memory sau này

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = run_pipeline(request.user_input, request.user_id)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))