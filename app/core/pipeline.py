# app/core/pipeline.py

from typing import Optional

from app.core.router import detect_intent
from app.llm.llm_client import generate_response
from app.tools.tool_registry import get_tool
from app.memory.memory_service import MemoryService
from app.core.strategy import TeachingStrategy
from app.rag.base_retriever import BaseRetriever


# ====================== Services ======================
memory_service = MemoryService()
strategy_engine = TeachingStrategy()
retriever: Optional[BaseRetriever] = None


def set_retriever(r: BaseRetriever):
    global retriever
    retriever = r


def run_pipeline(user_input: str, user_id: str = "default_user") -> str:
    """Pipeline hoàn chỉnh - Tuần 6"""

    # ========== 1. Load Memory ==========
    short_mem, long_mem = memory_service.load(user_id)

    target_lang = getattr(long_mem, "target_language", "pt-BR")
    teaching_lang = getattr(long_mem, "teaching_language", "vi")

    # ========== 2. Intent Detection ==========
    intent = detect_intent(user_input)

    # ========== 3. Strategy ==========
    strategy = strategy_engine.decide(
        intent=intent,
        long_mem=long_mem,
        short_mem=short_mem,
        user_input=user_input
    )

    final_result = None

    # ========== 4. Tool Handling ==========
    tool_func = get_tool(intent)

    if tool_func:
        try:
            if intent == "grammar":
                final_result = tool_func(
                    user_input,
                    lang=target_lang
                )

            elif intent == "translate":
                final_result = tool_func(
                    user_input,
                    target_lang=target_lang
                )

        except Exception as e:
            print(f"[Tool Error] {e}")

    # ========== 5. RAG Retrieval ==========
    rag_context = ""

    if retriever is not None:
        try:
            rag_results = retriever.retrieve(
                query=user_input,
                k=5,
                user_level=long_mem.level
            )

            if rag_results:
                rag_context = "\n\n".join(
                    item["text"] for item in rag_results
                )

        except Exception as e:
            print(f"[Retriever Error] {e}")

    # ========== 6. LLM Fallback ==========
    if not final_result:
        system_prompt = strategy_engine.build_system_prompt(strategy)

        final_result = generate_response(
            user_input=user_input,
            context={
                "system_prompt": system_prompt,
                "rag_context": rag_context,
                "history": short_mem.get_history()[-4:],
                "teaching_lang": teaching_lang,
                "target_lang": target_lang,
                "user_level": long_mem.level
            }
        )

    # ========== 7. Update Memory ==========
    memory_service.update_after_response(
        user_id=user_id,
        user_input=user_input,
        assistant_response=final_result,
        intent=intent
    )

    return final_result
