from app.tools.grammar_checker import run as grammar_run
from app.tools.translator import run as translator_run

# Đăng ký tool
TOOLS = {
    "grammar": grammar_run,
    "translate": translator_run,
}

def get_tool(tool_name: str):
    """Lấy tool theo tên"""
    return TOOLS.get(tool_name)