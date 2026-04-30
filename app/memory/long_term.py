# app/memory/long_term.py
from typing import Dict, List, Optional
from datetime import datetime

class LongTermMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.level: str = "A1"                    # CEFR level
        self.current_teaching_lang: str = "pt"
        self.error_count: int = 0
        self.common_errors: List[str] = []
        self.learned_vocabulary: List[str] = []
        self.goals: List[str] = []
        self.last_active: str = datetime.now().isoformat()
        self.total_sessions: int = 0

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "level": self.level,
            "current_teaching_lang": self.current_teaching_lang,
            "error_count": self.error_count,
            "common_errors": self.common_errors,
            "learned_vocabulary": self.learned_vocabulary,
            "goals": self.goals,
            "last_active": self.last_active,
            "total_sessions": self.total_sessions
        }

    @classmethod
    def from_dict(cls, data: Dict):
        mem = cls(data["user_id"])
        mem.level = data.get("level", "A1")
        mem.current_teaching_lang = data.get("current_teaching_lang", "pt-BR")
        mem.error_count = data.get("error_count", 0)
        mem.common_errors = data.get("common_errors", [])
        mem.learned_vocabulary = data.get("learned_vocabulary", [])
        mem.goals = data.get("goals", [])
        mem.last_active = data.get("last_active", datetime.now().isoformat())
        mem.total_sessions = data.get("total_sessions", 0)
        return mem