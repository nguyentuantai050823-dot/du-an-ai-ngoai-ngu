from typing import Dict, Any
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory

class TeachingStrategy:
    """Engine quyết định chiến thuật dạy học - Tuần 5 Final"""

    def decide(self, intent: str, long_mem: LongTermMemory, 
               short_mem: ShortTermMemory, user_input: str = "") -> Dict[str, Any]:
        
        # 1. Thu thập dữ liệu từ Memory
        errors = getattr(long_mem, "error_count", 0)
        total = getattr(long_mem, "total_sessions", 0)
        error_rate = errors / total if total > 0 else 0
        
        level = getattr(long_mem, "level", "A1")
        teach_lang = getattr(long_mem, "teaching_language", "vi") 

        # Strategy mặc định
        strategy = {
            "mode": "chat_mode",
            "difficulty": level,
            "explain_in": teach_lang,
            "intensity": "normal",
            "priority": 0 
        }

        # 2. Logic Priority
        if intent == "translate":
            strategy.update({"mode": "example_mode", "priority": 1})

        if intent == "grammar":
            strategy.update({"mode": "correction_mode", "priority": 2})

        if errors >= 4 or error_rate > 0.35:
            strategy.update({
                "mode": "practice_mode", 
                "intensity": "high", 
                "priority": 3
            })

        if self._is_repeating_error(short_mem.get_history()[-6:]):
            strategy.update({
                "mode": "practice_mode", 
                "intensity": "high", 
                "priority": 4
            })

        return strategy

    def _is_repeating_error(self, history: list) -> bool:
        """Kiểm tra xem Assistant có đang phải sửa lỗi liên tục không"""
        if len(history) < 4: return False
        count = sum(1 for msg in history if msg.get("role") == "assistant" 
                   and ("✅" in msg.get("content", "") or "Sửa lỗi:" in msg.get("content", "")))
        return count >= 2

    # PHẢI THỤT LỀ VÀO ĐÂY (4 dấu cách)
    def build_system_prompt(self, strategy: Dict[str, Any]) -> str:
        """Tạo lệnh đặc biệt cho LLM (Đã sửa lỗi lề và biến target)"""
        m = strategy.get("mode", "chat_mode")
        l = strategy.get("explain_in", "vi")
        target = strategy.get("target_lang", "ngôn ngữ mục tiêu")
        
        base = f"Role: Language Tutor. BẮT BUỘC giải thích bằng tiếng {l}. Teach in: {l}. Tone: {'Strict' if strategy.get('intensity') == 'high' else 'Friendly'}."

        if m == "practice_mode":
            return f"{base} Mode: PRACTICE. Chỉ đưa 1 câu hỏi {target} mức A1. KHÔNG giải thích."

        # 2. Chế độ Dịch & Ví dụ (Example Mode bạn đang tìm đây)
        elif m == "example_mode":
            return f"{base} Mode: EXAMPLE. Nhiệm vụ: Dịch câu sang {target}. Cho thêm 2 ví dụ ngắn dùng từ vựng trong câu đó. Giải thích từ mới bằng tiếng {l}."

        # 3. Chế độ Sửa lỗi
        elif m == "correction_mode":
            return f"{base} Mode: CORRECTION. Dùng ✅/❌ để sửa lỗi ngữ pháp. KHÔNG bịa chuyện ngoài lề. Giải thích ngắn bằng tiếng {l}."

        # 4. Chế độ Giao tiếp (Mặc định)
        else:
            return f"""{base} 
            Mode: CHAT. Nhiệm vụ: Trò chuyện bằng {target}. Yêu cầu:  Sử dụng từ vựng và cấu trúc ngữ pháp PHÙ HỢP với trình độ {user_level}, nếu {user_level} là A1-A2: Dùng câu ngắn, đơn giản, nếu {user_level} là B1-B2: Dùng câu phức, thành ngữ,  nếu user dùng tiếng {l}, hãy trả lời kèm bản dịch sang {target}."""