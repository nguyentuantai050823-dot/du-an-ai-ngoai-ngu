# app/memory/short_term.py
from typing import List, Dict
from datetime import datetime

class ShortTermMemory:
    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages
        self.messages: List[Dict] = []

    def add(self, role: str, content: str):
        """Thêm tin nhắn vào lịch sử"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Giữ chỉ max_messages tin gần nhất
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_history(self) -> List[Dict]:
        return self.messages

    def clear(self):
        self.messages.clear()