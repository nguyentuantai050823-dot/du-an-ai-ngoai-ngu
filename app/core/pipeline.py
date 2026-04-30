from app.core.router import detect_intent
from app.llm.llm_client import generate_response
from app.tools.tool_registry import get_tool
from app.memory.memory_service import MemoryService
from app.core.strategy import TeachingStrategy

# Khởi tạo Services
memory_service = MemoryService()
strategy_engine = TeachingStrategy()

def run_pipeline(user_input: str, user_id: str = "default_user") -> str:
    """Pipeline chính - Bản cập nhật Tuần 5 Chuẩn"""

    # 1. Load Memory (Long-term & Short-term)
    short_mem, long_mem = memory_service.load(user_id)
    
    # Lấy thông tin cấu hình từ Memory
    target_l = getattr(long_mem, "target_language", "pt-BR")
    teach_l = getattr(long_mem, "teaching_language", "vi")

    # 2. Nhận diện ý định (Intent)
    intent = detect_intent(user_input)

    # 3. Chiến thuật dạy học (Strategy)
    strategy = strategy_engine.decide(intent, long_mem, short_mem, user_input)

    # 4. Xử lý qua Tool (Nếu có)
    tool_func = get_tool(intent)
    final_result = None

    if tool_func:
        try:
            # FIX: Khớp nối Interface với các Tool cũ (Tuần 3)
            if intent == "grammar":
                tool_output = tool_func(user_input, lang=target_l) # Truyền lang thay vì context
            elif intent == "translate":
                tool_output = tool_func(user_input, target_lang=target_l)
            else:
                # Dự phòng cho tool mới nhận context
                tool_output = tool_func(user_input, context={"target_lang": target_l})
            
            final_result = tool_output if isinstance(tool_output, str) else tool_output.get("text")
        except Exception as e:
            print(f"[Pipeline Tool Error]: {e}")

    # 5. LLM Fallback (Xử lý bằng trí tuệ nhân tạo + Strategy)
    if not final_result:
        # Lấy prompt động từ chiến thuật
        system_prompt = strategy_engine.build_system_prompt(strategy)

        final_result = generate_response(
            user_input=user_input,
            context={
                "system_prompt": system_prompt, # Truyền lệnh dạy học vào LLM
                "history": short_mem.get_history(),
                "teaching_lang": teach_l,
                "target_lang": target_l,
                "user_level": long_mem.level
            }
        )

    # 6. Cập nhật Memory (Quan trọng để duy trì bộ não Strategy)
    memory_service.update_after_response(
        user_id=user_id,
        user_input=user_input,
        assistant_response=final_result,
        intent=intent
    )

    return final_result