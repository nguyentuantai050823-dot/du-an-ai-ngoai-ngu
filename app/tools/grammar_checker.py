import language_tool_python
from typing import Dict

# Cache
_tool_cache: Dict[str, language_tool_python.LanguageTool] = {}

# Danh sách ngôn ngữ hỗ trợ grammar checker (dễ thêm sau này)
SUPPORTED_GRAMMAR = {
    "pt": "pt-BR",   # Portuguese (Brazil)
    "en": "en-US",   # English
    "vi": "vi-VN",   # Vietnamese
}

def normalize_grammar_lang(lang_code: str) -> str:
    """Chuyển ngôn ngữ ngắn thành code đầy đủ cho LanguageTool"""
    lang_lower = lang_code.lower()
    if lang_lower in SUPPORTED_GRAMMAR:
        return SUPPORTED_GRAMMAR[lang_lower]
    if lang_lower in SUPPORTED_GRAMMAR.values():
        return lang_lower
    return "pt-BR"   # default hiện tại là tiếng Bồ Đào Nha Brazil


def get_grammar_tool(lang: str = "pt") -> language_tool_python.LanguageTool:
    lang_code = normalize_grammar_lang(lang)
    
    if lang_code not in _tool_cache:
        try:
            print(f"[INFO] Khởi tạo LanguageTool cho {lang_code}")
            _tool_cache[lang_code] = language_tool_python.LanguageTool(lang_code)
        except Exception:
            _tool_cache[lang_code] = language_tool_python.LanguageToolPublicAPI(lang_code)
    
    return _tool_cache[lang_code]


def run(user_input: str, lang: str = "pt") -> str:
    """Sửa lỗi ngữ pháp - giải thích bằng tiếng Việt"""
    if not user_input or not user_input.strip():
        return "Bạn chưa nhập câu nào để kiểm tra."

    try:
        tool = get_grammar_tool(lang)
        matches = tool.check(user_input)

        if not matches:
            return f"✅ Câu của bạn đã đúng ngữ pháp!\n\nCâu gốc: {user_input}"

        corrected_text = language_tool_python.utils.correct(user_input, matches)

        result = f"✅ **Câu đã sửa:** {corrected_text}\n\n"
        result += "**Các lỗi được phát hiện:**\n"

        for i, match in enumerate(matches, 1):
            suggestion = match.replacements[0] if match.replacements else "Không có gợi ý"
            result += f"{i}. {match.message}\n   → Gợi ý: {suggestion}\n"

        return result

    except Exception as e:
        return f"❌ Lỗi khi kiểm tra ngữ pháp: {str(e)}"