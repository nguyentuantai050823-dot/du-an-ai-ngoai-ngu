import numpy as np
from typing import List, Dict, Any

from .base_retriever import BaseRetriever


class DF_Retriever_Lite(BaseRetriever):
    """
    DF-RAG Lite - Tuần 6

    Features:
    - FAISS initial retrieval
    - geometric MMR
    - multi-lambda selection
    - lightweight scoring
    """

    def __init__(self, faiss_index, id_to_chunk: Dict[int, Dict]):
        self.faiss_index = faiss_index
        self.id_to_chunk = id_to_chunk
        self.lambdas = [0.3, 0.5, 0.7, 0.85, 1.0]

    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks"""

        if k <= 0:
            k = 5

        query_emb = self._get_query_embedding(query)

        distances, indices = self.faiss_index.search(
            query_emb.reshape(1, -1),
            40
        )

        candidates = []

        for idx in indices[0]:
            if idx in self.id_to_chunk:
                candidates.append(self.id_to_chunk[idx])

        if not candidates:
            return []

        all_embs = [
            np.array(c["embedding"], dtype=np.float32)
            for c in candidates
        ]

        all_texts = [
            c["text"]
            for c in candidates
        ]

        all_metadata = [
            c.get("metadata", {})
            for c in candidates
        ]

        return self._select_best_lambda(
            query_emb,
            all_embs,
            all_texts,
            all_metadata,
            k
        )

    def _get_query_embedding(self, query: str) -> np.ndarray:
        from app.rag.embedder import get_embedding

        emb = get_embedding(query)
        return np.array(emb, dtype=np.float32)

    def _select_best_lambda(
        self,
        query_emb,
        all_embs,
        all_texts,
        all_metadata,
        k
    ):
        best_score = -np.inf
        best_result = []

        for lam in self.lambdas:
            selected = self._gmmr_select(
                query_emb,
                all_embs,
                all_texts,
                all_metadata,
                lam,
                k
            )

            score = sum(c["score"] for c in selected)

            if score > best_score:
                best_score = score
                best_result = selected

        return best_result

    def _gmmr_select(
        self,
        query_emb,
        all_embs,
        all_texts,
        all_metadata,
        lambda_val,
        k
    ):
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        embs_norm = [
            e / (np.linalg.norm(e) + 1e-8)
            for e in all_embs
        ]

        selected_idx = []
        selected_embs = []

        for _ in range(min(k, len(embs_norm))):
            best_score = -np.inf
            best_i = -1

            for i, emb in enumerate(embs_norm):
                if i in selected_idx:
                    continue

                score = self._gmmr_score(
                    q,
                    emb,
                    selected_embs,
                    lambda_val
                )

                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i == -1:
                break

            selected_idx.append(best_i)
            selected_embs.append(embs_norm[best_i])

        return [
            {
                "text": all_texts[i],
                "metadata": all_metadata[i],
                "score": float(np.dot(q, embs_norm[i]))
            }
            for i in selected_idx
        ]

    def _gmmr_score(
        self,
        query_vec,
        candidate_vec,
        selected_vecs,
        lambda_val
    ):
        relevance = float(np.dot(query_vec, candidate_vec))

        if not selected_vecs:
            return relevance

        centroid = np.mean(selected_vecs, axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)

        diversity = np.sqrt(
            max(
                0.0,
                2 - 2 * float(np.dot(candidate_vec, centroid))
            )
        )

        return lambda_val * relevance + (1 - lambda_val) * diversity