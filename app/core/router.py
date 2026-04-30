def detect_intent(user_input: str) -> str:
    """Intent detection đơn giản - sau này nâng cấp thành LLM-based"""
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["dịch", "translate", "traduzir"]):
        return "translate"
    if any(word in user_lower for word in ["sửa", "chữa", "fix", "grammar", "corrigir"]):
        return "grammar"
    if any(word in user_lower for word in ["bài tập", "exercise", "practice", "exercício", "praticar"]):
        return "exercise"
    
    return "chat"