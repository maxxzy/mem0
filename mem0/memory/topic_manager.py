import os
from typing import Dict, Set, List, Optional, Tuple
import json
import numpy as np

class TopicManager:
    def __init__(self, savePath: str, embeddingDim: int = 1536, topics_max_num: int = 1000):
        self.savePath = savePath
        os.makedirs(os.path.dirname(self.savePath), exist_ok=True)
        self.embeddingDim = embeddingDim
        self.topics: Dict[str, Dict] = {}
        self.inverted: Dict[str, Set[str]] = {}
        self.count = 0
        self.topics_max_num = topics_max_num

    def assign_memory_to_topic(self, memory_id: str, topics: List[str], topic_embeddings: Optional[Dict[str, np.ndarray]] = None):
        if topic_embeddings is None:
            topic_embeddings = {}

        for t in topics:
            if t not in self.inverted:
                self.inverted[t] = set()
            self.inverted[t].add(memory_id)

            if t not in self.topics:
                self.topics[t] = {
                    "embedding": None,
                    "count": 0,
                }

            self.topics[t]["count"] += 1

            if t in topic_embeddings and topic_embeddings[t] is not None:
                emb = np.asarray(topic_embeddings[t], dtype=np.float32)
                prev = self.topics[t]["embedding"]
                cnt = self.topics[t]["count"]
                if prev is None:
                    self.topics[t]["embedding"] = emb
                else:
                    self.topics[t]["embedding"] = prev + (emb - prev) / float(cnt)

        if self.count > self.topics_max_num:
            self.merge_topics(similarity_threshold=0.85)

    def _cosine_sim(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return -1.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def get_relevant_topics(self,
                            query_topics: List[str],
                            top_k: int = 5,
                            similarity_threshold: float = 0.75,
                            query_embedding: Optional[np.ndarray] = None) -> List[str]:
        base_hits = [t for t in query_topics if t in self.inverted and len(self.inverted[t]) > 0]
        candidates = [(t, meta.get("embedding")) for t, meta in self.topics.items() if isinstance(meta.get("embedding"), np.ndarray)]
        if not candidates:
            return sorted(set(base_hits))

        topic_to_emb = {t: emb for t, emb in candidates}

        if isinstance(query_embedding, np.ndarray):
            qe = np.asarray(query_embedding, dtype=np.float32)
            if qe.shape[-1] != self.embeddingDim:
                raise ValueError(f"Query embedding dim mismatch: expected {self.embeddingDim}, got {qe.shape[-1]}")
            sims = []
            for t, emb in topic_to_emb.items():
                if len(self.inverted.get(t, set())) == 0:
                    continue
                sim = self._cosine_sim(qe, emb)
                if sim >= similarity_threshold:
                    sims.append((t, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            top = [t for t, _ in sims[:max(1, top_k)]]
            return sorted(set(top) | set(base_hits))
        '''
        result = []
        for qt in query_topics:
            qt_emb = topic_to_emb.get(qt, None)
            if qt_emb is None:
                if include_original and qt in self.inverted and len(self.inverted[qt]) > 0:
                    result.append(qt)
                continue

            sims = []
            for t, emb in topic_to_emb.items():
                if len(self.inverted.get(t, set())) == 0:
                    continue
                sim = self._cosine_sim(qt_emb, emb)
                if sim >= similarity_threshold:
                    sims.append((t, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            top = [t for t, _ in sims[:max(1, top_k)]]

            if include_original and qt in self.inverted and len(self.inverted[qt]) > 0:
                top.append(qt)
            result.extend(top)

        return sorted(set(result))
        '''
        return set(base_hits)

    def get_memory_ids_for_topics(self, topics: List[str]) -> Set[str]:
        mem_ids: Set[str] = set()
        for t in topics:
            mem_ids |= self.inverted.get(t, set())
        return mem_ids

    def _merge_topic_into(self, src: str, dst: str) -> None:
        if src == dst:
            return
        if src not in self.topics or dst not in self.topics:
            return

        src_set = self.inverted.get(src, set())
        dst_set = self.inverted.get(dst, set())
        dst_set |= src_set
        self.inverted[dst] = dst_set
        if src in self.inverted:
            del self.inverted[src]

        src_meta = self.topics[src]
        dst_meta = self.topics[dst]
        ci = int(dst_meta.get("count", 0))
        cj = int(src_meta.get("count", 0))
        dst_meta["count"] = ci + cj

        ei = dst_meta.get("embedding", None)
        ej = src_meta.get("embedding", None)
        if isinstance(ei, np.ndarray) and isinstance(ej, np.ndarray):
            total = max(ci + cj, 1)
            dst_meta["embedding"] = (ei * ci + ej * cj) / float(total)
        elif ei is None and isinstance(ej, np.ndarray):
            dst_meta["embedding"] = ej

        del self.topics[src]

    def merge_topics(self, similarity_threshold: float = 0.85):
        candidates = [t for t, meta in self.topics.items() if isinstance(meta.get("embedding"), np.ndarray)]
        remaining = set(candidates)

        changed = True
        while changed:
            changed = False
            to_remove = set()
            topics_list = sorted(list(remaining))
            n = len(topics_list)
            for i in range(n):
                ti = topics_list[i]
                if ti in to_remove:
                    continue
                for j in range(i + 1, n):
                    tj = topics_list[j]
                    if tj in to_remove:
                        continue
                    si = self.topics[ti]["embedding"]
                    sj = self.topics[tj]["embedding"]
                    sim = self._cosine_sim(si, sj)
                    if sim >= similarity_threshold:
                        ci = self.topics[ti]["count"]
                        cj = self.topics[tj]["count"]
                        if ci > cj or (ci == cj and ti < tj):
                            dst, src = ti, tj
                        else:
                            dst, src = tj, ti

                        self._merge_topic_into(src, dst)
                        to_remove.add(src)
                        changed = True
            remaining -= to_remove

    def _get_topics_by_mem_id(self, memory_id: str) -> Set[str]:
        current: Set[str] = set()
        for t, members in self.inverted.items():
            if memory_id in members:
                current.add(t)
        return current

    def update_topic_by_mem_id(
        self,
        memory_id: str,
        topics: List[str],
        topic_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ):
        if topic_embeddings is None:
            topic_embeddings = {}

        new_topics: Set[str] = set(topics or [])
        old_topics: Set[str] = self._get_topics_by_mem_id(memory_id)

        to_add = new_topics - old_topics
        to_remove = old_topics - new_topics

        for t in to_add:
            self.inverted.setdefault(t, set()).add(memory_id)
            if t not in self.topics:
                self.topics[t] = {"embedding": None, "count": 0}
            self.topics[t]["count"] = int(self.topics[t].get("count", 0)) + 1

            emb = topic_embeddings.get(t)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                prev = self.topics[t].get("embedding", None)
                cnt = int(self.topics[t]["count"])
                if prev is None:
                    self.topics[t]["embedding"] = emb
                else:
                    self.topics[t]["embedding"] = prev + (emb - prev) / float(cnt)

        for t in to_remove:
            members = self.inverted.get(t)
            if members is not None and memory_id in members:
                members.remove(memory_id)
                if len(members) == 0:
                    self.inverted[t] = members
            if t in self.topics:
                cur = int(self.topics[t].get("count", 0))
                self.topics[t]["count"] = max(cur - 1, 0)

        if self.count > self.topics_max_num:
            self.merge_topics(similarity_threshold=0.85)

    def delete_memory(self, memory_id: str):
        current_topics = self._get_topics_by_mem_id(memory_id)
        for t in current_topics:
            members = self.inverted.get(t)
            if members is not None and memory_id in members:
                members.remove(memory_id)
                if t in self.topics:
                    cur = int(self.topics[t].get("count", 0))
                    self.topics[t]["count"] = max(cur - 1, 0)

                if len(members) == 0:
                    self.inverted.pop(t, None)
                    if t in self.topics and int(self.topics[t].get("count", 0)) == 0:
                        self.topics.pop(t, None)

    def save():
        pass

    def load():
        pass