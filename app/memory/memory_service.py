# app/memory/memory_service.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from .short_term import ShortTermMemory
from .long_term import LongTermMemory


class MemoryService:
    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.short_term: Dict[str, ShortTermMemory] = {}
        self.long_term: Dict[str, LongTermMemory] = {}

    def _get_long_term_path(self, user_id: str) -> Path:
        """Tạo tên file an toàn từ user_id (email)"""
        safe_id = user_id.replace("@", "_at_").replace(".", "_").replace("+", "_")
        return self.storage_dir / f"{safe_id}_long.json"

    def load(self, user_id: str) -> Tuple[ShortTermMemory, LongTermMemory]:
        """Load memory cho user"""
        # Long-term memory
        if user_id not in self.long_term:
            path = self._get_long_term_path(user_id)
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.long_term[user_id] = LongTermMemory.from_dict(data)
                except Exception as e:
                    print(f"[Memory] Lỗi đọc file long-term cho {user_id}: {e}")
                    self.long_term[user_id] = LongTermMemory(user_id)
            else:
                self.long_term[user_id] = LongTermMemory(user_id)

        # Short-term memory
        if user_id not in self.short_term:
            self.short_term[user_id] = ShortTermMemory(max_messages=15)

        return self.short_term[user_id], self.long_term[user_id]

    def save_long_term(self, user_id: str):
        """Lưu long-term memory xuống JSON"""
        if user_id in self.long_term:
            try:
                path = self._get_long_term_path(user_id)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.long_term[user_id].to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Memory] Lỗi lưu long-term cho {user_id}: {e}")

    def update_after_response(
        self, 
        user_id: str, 
        user_input: str, 
        assistant_response: str, 
        intent: str = "chat"
    ):
        """Cập nhật memory sau khi trả lời user - Phiên bản Tuần 5"""
        short_mem, long_mem = self.load(user_id)
        
        # 1. Cập nhật Short-term (Lịch sử chat)
        short_mem.add("user", user_input)
        short_mem.add("assistant", assistant_response)

        # 2. Cập nhật Long-term
        long_mem.last_active = datetime.now().isoformat()
        
        # BẮT BUỘC: Tăng total_sessions để bộ não Strategy có mẫu số tính toán
        # Ở level này, ta coi mỗi lượt tương tác là một đơn vị dữ liệu để đánh giá error_rate
        long_mem.total_sessions += 1

        # 3. Logic đếm lỗi (Cực kỳ quan trọng cho Strategy)
        # Nếu AI đang ở chế độ sửa lỗi hoặc intent là grammar
        is_error_detected = any(marker in assistant_response for marker in ["❌", "Sửa lỗi:", "Incorrect:"])
        
        if intent == "grammar" or is_error_detected:
            # Chỉ đếm lỗi nếu Assistant thực sự chỉ ra chỗ sai (tránh đếm nhầm khi AI khen)
            if any(kw in assistant_response.lower() for kw in ["sai", "lỗi", "mistake", "wrong"]):
                long_mem.error_count += 1
                
                # [Nâng cấp nhẹ]: Lưu lại từ vựng/cấu trúc hay sai nếu có thể
                # Ví dụ: long_mem.common_errors.append(user_input) 

        # 4. Lưu xuống file JSON
        self.save_long_term(user_id)