import time
import ollama

MAX_EMBEDDING_CHARS = 1500


class EmbeddingModel:
    def __init__(self, model_name="bge-m3:latest", max_retries=2, retry_delay=0.5):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_embedding(self, text):
        """Biến một chuỗi văn bản thành một vector số (list of floats)"""
        text = text.strip()
        if not text:
            return []

        if len(text) > MAX_EMBEDDING_CHARS:
            text = text[:MAX_EMBEDDING_CHARS]

        for attempt in range(self.max_retries + 1):
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                embedding = response.get("embedding", [])
                if not isinstance(embedding, list) or not embedding:
                    return []
                return embedding
            except Exception as e:
                if attempt < self.max_retries:
                    sleep_s = self.retry_delay * (2 ** attempt)
                    print(f"⚠️ Embedding error (retry {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(sleep_s)
                else:
                    print(f"⚠️ Embedding error: {e}")
                    return []

    def get_embeddings(self, texts):
        """Biến danh sách văn bản thành danh sách vector theo từng item."""
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
        return embeddings

    def get_dimension(self):
        """Lấy dimension embedding hiện tại từ model."""
        test_emb = self.get_embedding("Test dimension detection.")
        if isinstance(test_emb, list) and len(test_emb) > 0:
            return len(test_emb)
        return None


if __name__ == "__main__":
    embedder = EmbeddingModel()
    vector = embedder.get_embedding("Xin chào, tôi đang học tiếng Brazil")
    print(f"Độ dài vector: {len(vector)}")