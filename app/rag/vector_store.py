import faiss
import numpy as np
import pickle
import os
from typing import Optional


class VectorStore:
    def __init__(self, dimension: Optional[int] = None, persist_path=None):
        # dimension có thể None, sẽ được suy ra từ vector đầu tiên hoặc từ index đã load
        self.dimension = dimension

        if persist_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_path = os.path.join(base_dir, "data", "vector_store")
        self.persist_path = persist_path

        self.index = None
        self.metadata = []
        self._id_set = set()  # ← Track ID để kiểm tra duplicate nhanh O(1)

        self.load()  

    def has_id(self, id: str) -> bool:
        """Kiểm tra ID đã tồn tại chưa — dùng để bỏ qua embedding tốn kém"""
        return id in self._id_set

    def add(self, id: str, vector, text: str, metadata: dict) -> bool:
        """
        Thêm vector mới. Trả về True nếu thêm thành công,
        False nếu ID đã tồn tại (bỏ qua) hoặc dimension không khớp.
        """
        if id in self._id_set:
            return False

        vec = np.asarray(vector, dtype="float32")
        if vec.ndim != 1 or vec.size == 0:
            return False

        current_dim = int(vec.shape[0])
        if not self._ensure_index_for_dim(current_dim, context=f"id={id}"):
            return False

        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)

        self.index.add(vec)
        self.metadata.append({
            "id": id,
            "text": text,
            "metadata": metadata
        })
        self._id_set.add(id)
        return True

    def add_batch(self, ids, vectors, texts, metadatas) -> int:
        """
        Thêm nhiều vector một lần. Trả về số lượng vector add thành công.
        Dùng numpy + normalize theo lô để tăng tốc rõ rệt.
        """
        accepted_ids = []
        accepted_texts = []
        accepted_metadatas = []
        accepted_vectors = []

        for id_, vec, text, meta in zip(ids, vectors, texts, metadatas):
            if id_ in self._id_set:
                continue
            if not vec:
                continue
            arr = np.asarray(vec, dtype="float32")
            if arr.ndim != 1 or arr.size == 0:
                continue
            accepted_ids.append(id_)
            accepted_texts.append(text)
            accepted_metadatas.append(meta)
            accepted_vectors.append(arr)

        if not accepted_vectors:
            return 0

        vecs = np.vstack(accepted_vectors).astype("float32")
        current_dim = int(vecs.shape[1])

        if not self._ensure_index_for_dim(current_dim):
            return 0

        faiss.normalize_L2(vecs)
        self.index.add(vecs)

        for id_, text, meta in zip(accepted_ids, accepted_texts, accepted_metadatas):
            self.metadata.append({
                "id": id_,
                "text": text,
                "metadata": meta
            })
            self._id_set.add(id_)

        return len(accepted_ids)

    def save(self):
        os.makedirs(self.persist_path, exist_ok=True)
        if self.index is None:
            print("⚠️ Vector store is empty, skip save.")
            return

        faiss.write_index(
            self.index,
            os.path.join(self.persist_path, "index.faiss")
        )

        with open(os.path.join(self.persist_path, "store.pkl"), "wb") as f:
            pickle.dump({
                "metadata": self.metadata
            }, f)

        print(f"💾 Saved vector store ({len(self.metadata)} vectors)")

    def _ensure_index_for_dim(self, current_dim: int, context: str = "batch") -> bool:
        """
        Đảm bảo index và dimension nhất quán trước khi add vector.
        """
        if self.index is None:
            if self.dimension is None:
                self.dimension = current_dim
            elif current_dim != self.dimension:
                print(f"⚠️ Skip {context}: dim {current_dim} != expected {self.dimension}")
                return False
            self.index = faiss.IndexFlatIP(self.dimension)
            return True

        if current_dim != self.dimension:
            print(f"⚠️ Skip {context}: dim {current_dim} != expected {self.dimension}")
            return False
        return True

    def load(self):
        index_path = os.path.join(self.persist_path, "index.faiss")
        store_path = os.path.join(self.persist_path, "store.pkl")

        if os.path.exists(index_path) and os.path.exists(store_path):
            self.index = faiss.read_index(index_path)
            # Cập nhật lại dimension từ index đã lưu
            self.dimension = self.index.d

            with open(store_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]

            # ← Rebuild _id_set từ dữ liệu đã load
            self._id_set = {item["id"] for item in self.metadata}

            print(f"✅ Loaded vector store ({len(self.metadata)} vectors)")
        else:
            print("⚠️ No existing vector store found. Starting fresh.")