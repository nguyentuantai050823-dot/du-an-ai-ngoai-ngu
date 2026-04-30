# app/tools/translator.py

from deep_translator import GoogleTranslator

LANGUAGE_NAMES = {
    "pt": "Tiếng Bồ Đào Nha (Brazil)",
    "vi": "Tiếng Việt",
    "en": "Tiếng Anh",
}

def detect_target_lang(text: str) -> str:
    """Phát hiện ngôn ngữ đích một cách thông minh"""
    text_lower = text.lower().strip()

    if "dịch sang" in text_lower:
        if any(k in text_lower for k in ["tiếng anh", "english", "anh"]):
            return "en"
        if any(k in text_lower for k in ["tiếng việt", "vietnamese", "việt"]):
            return "vi"
        if any(k in text_lower for k in ["tiếng bồ","tiếng brazil", "portuguese", "brazil"]):
            return "pt"

    if text_lower.startswith(("translate:", "translate ")):
        return "en"
    if text_lower.startswith(("traduzir:", "traduzir ")):
        return "pt"
    if text_lower.startswith(("dịch:", "dịch ")):
        return "vi"

    return "vi"


def extract_text(user_input: str) -> str:
    if ":" in user_input:
        return user_input.split(":", 1)[1].strip()
    return user_input.strip()


# app/tools/translator.py

def run(user_input: str, target_lang: str = None) -> str:
    """Translator Tool"""
    try:
        if not user_input or not user_input.strip():
            return "Bạn chưa nhập nội dung cần dịch."

        # Ưu tiên 1: Nếu pipeline truyền target_lang vào → dùng nó
        # Ưu tiên 2: Detect từ câu nói của user
        if target_lang is None:
            target_lang = detect_target_lang(user_input)

        text_to_translate = extract_text(user_input)

        if not text_to_translate:
            return "Không tìm thấy nội dung cần dịch."

        translated = GoogleTranslator(source='auto', target=target_lang).translate(text_to_translate)

        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())

        return f"🔄 **Dịch sang {lang_name}:**\n{translated}"

    except Exception as e:
        return f"❌ Lỗi khi dịch: {str(e)}\nVui lòng thử lại."