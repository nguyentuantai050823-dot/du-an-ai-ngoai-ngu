import ollama
from app.llm.prompts import get_system_prompt

def generate_response(user_input: str, context: dict, model: str = "qwen2.5:3b") -> str:
    """
    Hàm gọi LLM đã được nâng cấp để nhận context (Tuần 4)
    """
    try:
        # 1. Lấy các thông tin từ context (với giá trị mặc định để tránh lỗi)
        history = context.get("history", [])
        teaching_lang = context.get("teaching_lang", "vi")
        target_lang = context.get("target_lang", "pt-BR")
        user_level = context.get("user_level", "A1")
        rag_context = context.get("rag_context", "")

        # 2. Chuẩn bị danh sách tin nhắn (Messages)
        # Bắt đầu bằng System Prompt (đã cá nhân hóa theo ngôn ngữ dạy)
        system_prompt = context.get("system_prompt", get_system_prompt(teaching_lang, target_lang, user_level))

        messages = [{"role": "system", "content": system_prompt}]

        # 3. Thêm lịch sử trò chuyện (Short-term memory)
        # Duyệt qua history và thêm vào messages theo đúng định dạng role: user/assistant
        for msg in history:
            messages.append(msg)
        
        # 3.5 Thêm RAG context nếu có
        if rag_context:
           messages.append({
               "role": "system",
               "content": f"Knowledge Context:\n{rag_context}"
        })

        # 4. Thêm câu hỏi hiện tại của User
        messages.append({"role": "user", "content": user_input})

        # 5. Gọi Ollama
        response = ollama.chat(
            model=model,
            messages=messages
        )
        
        return response['message']['content']

    except Exception as e:
        return f"Lỗi khi gọi Ollama: {str(e)}. Đảm bảo Ollama đang chạy."