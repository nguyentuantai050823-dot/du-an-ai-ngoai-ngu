import os
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set

class GraphStore:
    def __init__(self, persist_path: Optional[str] = None):
        if persist_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_path = os.path.join(base_dir, "data", "graph_store", "graph.json")
        
        self.persist_path = persist_path
        self.triples: List[Dict] = []
        self._triple_set: Set[str] = set()
        
        # Indexes
        self._subject_index: Dict[str, List[int]] = defaultdict(list)
        self._relation_index: Dict[str, List[int]] = defaultdict(list)
        self._object_index: Dict[str, List[int]] = defaultdict(list)   # ← ĐÃ THÊM

        self.load()

    @staticmethod
    def _make_key(s: str, r: str, o: str, sid: str) -> str:
        return f"{s.strip().lower()}|{r.strip().lower()}|{o.strip().lower()}|{sid}"

    def add_triple(self, subject: str, relation: str, obj: str, provenance: Dict) -> bool:
        key = self._make_key(subject, relation, obj, provenance["sentence_id"])
        if key in self._triple_set:
            return False

        triple_dict = {
            "subject": subject.strip(),
            "relation": relation.strip(),
            "object": obj.strip(),
            "sentence_id": provenance["sentence_id"],
            "passage_id": provenance.get("passage_id"),
            "metadata": provenance.get("metadata", {})
        }

        idx = len(self.triples)
        self.triples.append(triple_dict)
        self._triple_set.add(key)

        # Update indexes
        s_lower = subject.strip().lower()
        r_lower = relation.strip().lower()
        o_lower = obj.strip().lower()
        
        self._subject_index[s_lower].append(idx)
        self._relation_index[r_lower].append(idx)
        self._object_index[o_lower].append(idx)

        return True

    def save(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self.triples, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu Graph Store: {len(self.triples):,} triples")

    def load(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self.triples = json.load(f)

            self._triple_set.clear()
            self._subject_index.clear()
            self._relation_index.clear()
            self._object_index.clear()

            for idx, t in enumerate(self.triples):
                key = self._make_key(t["subject"], t["relation"], t["object"], t["sentence_id"])
                self._triple_set.add(key)
                self._subject_index[t["subject"].lower()].append(idx)
                self._relation_index[t["relation"].lower()].append(idx)
                self._object_index[t["object"].lower()].append(idx)

            print(f"✅ Đã load Graph Store: {len(self.triples):,} triples")
        else:
            print("⚠️ Chưa có graph.json → khởi tạo mới.")

    def get_stats(self) -> Dict:
        return {
            "total_triples": len(self.triples),
            "unique_subjects": len(self._subject_index),
            "unique_relations": len(self._relation_index),
            "unique_objects": len(self._object_index),
            "file_size_kb": round(os.path.getsize(self.persist_path) / 1024, 2) if os.path.exists(self.persist_path) else 0
        }

    # ====================== SEARCH TỐI ƯU ======================
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Token-based + scoring (không còn substring thô)"""
        query_tokens = set(query.strip().lower().split())
        if not query_tokens:
            return []

        scored = []
        for idx, t in enumerate(self.triples):
            subj = t["subject"].lower()
            rel = t["relation"].lower()
            obj = t["object"].lower()
            
            subj_tokens = set(subj.split())
            obj_tokens = set(obj.split())
            
            overlap = len(query_tokens & subj_tokens) * 3 + \
                      len(query_tokens & obj_tokens) * 2 + \
                      (1 if any(q in rel for q in query_tokens) else 0)
            
            if overlap > 0:
                scored.append((overlap, t))

        # Sort theo score giảm dần
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scored[:limit]]

    # ====================== TRAVERSAL ======================
    def get_neighbors(self, entity: str) -> List[Dict]:
        """O(1) nhờ cả 3 index"""
        entity = entity.strip().lower()
        indices = set(self._subject_index[entity] + self._object_index[entity])
        return [self.triples[i] for i in indices]

    def get_related(self, subject: str, limit: int = 30) -> List[Dict]:
        subject = subject.strip().lower()
        indices = self._subject_index.get(subject, [])
        return [self.triples[i] for i in indices[:limit]]

    def multi_hop(self, start_entity: str, hops: int = 2) -> Set[str]:
        """Multi-hop BFS – đây mới là sức mạnh GraphRAG"""
        visited = set()
        frontier = [start_entity.strip().lower()]

        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                neighbors = self.get_neighbors(node)
                for t in neighbors:
                    obj = t["object"].strip().lower()
                    if obj not in visited:
                        next_frontier.append(obj)
            frontier = next_frontier
        return visited

    def clear(self):
        self.triples.clear()
        self._triple_set.clear()
        self._subject_index.clear()
        self._relation_index.clear()
        self._object_index.clear()
        print("🧹 Graph Store đã được xóa sạch.")

    def export_to_jsonl(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for t in self.triples:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"📤 Exported {len(self.triples)} triples → {path}")