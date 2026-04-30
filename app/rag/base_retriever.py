from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Base Retriever Interface"""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Trả về list dict chứa chunk + metadata"""
        pass

    def get_context(self, query: str, k: int = 5, **kwargs) -> str:
        """Trả về chuỗi context sạch để đưa vào LLM"""
        results = self.retrieve(query, k, **kwargs)
        return "\n\n".join(item["text"] for item in results)
