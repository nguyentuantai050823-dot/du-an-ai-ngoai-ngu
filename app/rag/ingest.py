import os
import json
import hashlib
from typing import List

from tqdm import tqdm

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.graph_store import GraphStore


class Ingestor:
    def __init__(self, batch_size: int = 64, save_every: int = 500):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_root = os.path.join(self.base_dir, "data", "processed_data")

        self.embedding_engine = EmbeddingModel()

        dim = self.embedding_engine.get_dimension()
        if dim:
            print(f"🔢 Detected embedding dimension: {dim}")
            self.vector_db = VectorStore(dimension=dim)
        else:
            print("⚠️ Could not detect dimension, fallback to 1024")
            self.vector_db = VectorStore(dimension=1024)

        self.graph_db = GraphStore()

        self.batch_size = batch_size
        self.save_every = save_every
        self._vectors_since_save = 0

    def run_ingestion(self):
        json_files = self._get_json_files()

        print(f"🔍 Found {len(json_files)} JSON files to ingest.\n")

        total_vec_added = 0
        total_triple_added = 0

        for file_path in tqdm(json_files, desc="Processing files", unit="file"):
            rel_path = os.path.relpath(file_path, self.base_dir)
            print(f"📄 Processing: {rel_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            normalized = self._normalize_data_schema(data, file_path)

            vec_added = (
                self._ingest_vectors(normalized["passages"], "passage") +
                self._ingest_vectors(normalized["sentences"], "sentence")
            )
            triple_added = self._ingest_triples(normalized["triples"])

            total_vec_added += vec_added
            total_triple_added += triple_added

            print(f"✅ Done: +{vec_added} vectors, +{triple_added} triples\n")

        if total_vec_added > 0:
            self.vector_db.save()
        if total_triple_added > 0:
            self.graph_db.save()

        print("✨ Ingestion completed successfully!")
        print(f"   Total new vectors : {total_vec_added:,}")
        print(f"   Total new triples : {total_triple_added:,}")

    def _get_json_files(self) -> List[str]:
        json_files = []
        for root, _, files in os.walk(self.processed_root):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        return json_files

    def _normalize_data_schema(self, data, file_path: str) -> dict:
        rel_path = os.path.relpath(file_path, self.base_dir).lower()

        if isinstance(data, dict) and any(k in data for k in ["passages", "sentences"]):
            return {
                "passages": data.get("passages", []),
                "sentences": data.get("sentences", []),
                "triples": data.get("triples", []),
            }

        if isinstance(data, list):
            passages = []
            sentences = []

            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue

                text = (
                    item.get("text_to_embed")
                    or item.get("text")
                    or item.get("content")
                    or ""
                ).strip()

                if not text:
                    continue

                raw_id = item.get("id") or item.get("ID")
                if not raw_id:
                    seed = f"{rel_path}:{i}:{text[:250]}"
                    raw_id = hashlib.md5(seed.encode("utf-8")).hexdigest()

                metadata = item.get("metadata", {})
                if not metadata:
                    metadata = {}
                    if "page" in item:
                        metadata["page"] = item["page"]
                    if "pos" in item:
                        metadata["pos"] = item["pos"]

                entry = {
                    "id": str(raw_id),
                    "text": text,
                    "metadata": metadata,
                }

                if any(x in rel_path for x in ["grammar", "exercise", "lesson"]):
                    passages.append(entry)
                else:
                    sentences.append(entry)

            print(f"   Normalized → {len(passages)} passages | {len(sentences)} sentences")
            return {"passages": passages, "sentences": sentences, "triples": []}

        print(f"⚠️ Unsupported schema: {rel_path}")
        return {"passages": [], "sentences": [], "triples": []}

    def _ingest_vectors(self, items: list, granularity_type: str) -> int:
        added = 0
        pending_items = [
            item for item in items
            if not self.vector_db.has_id(item["id"]) and item.get("text", "").strip()
        ]

        if not pending_items:
            return 0

        for i in tqdm(range(0, len(pending_items), self.batch_size),
                      desc=f" ↳ Vectors ({granularity_type})", unit="batch", leave=False):
            batch = pending_items[i:i + self.batch_size]

            texts = [b["text"] for b in batch]
            ids = [b["id"] for b in batch]
            metadatas = [{**b.get("metadata", {}), "type": granularity_type} for b in batch]

            embeddings = self.embedding_engine.get_embeddings(texts)

            added_now = self.vector_db.add_batch(
                ids=ids, vectors=embeddings, texts=texts, metadatas=metadatas
            )

            added += added_now
            self._vectors_since_save += added_now

            if self._vectors_since_save >= self.save_every:
                self.vector_db.save()
                self._vectors_since_save = 0

        return added

    def _ingest_triples(self, triples: list) -> int:
        added = 0
        for t in tqdm(triples, desc="Triples", unit="triple", leave=False):
            if not all(k in t for k in ["subject", "relation", "object"]):
                continue

            was_added = self.graph_db.add_triple(
                subject=t["subject"],
                relation=t["relation"],
                object=t["object"],
                provenance={
                    "sentence_id": t.get("sentence_id", ""),
                    "passage_id": t.get("passage_id", "")
                }
            )
            if was_added:
                added += 1
        return added


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.run_ingestion()