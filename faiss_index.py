"""
Step 7: FAISS Matching Algorithm
Efficient similarity search for face embeddings.
Falls back to manual cosine similarity if faiss-cpu is not installed.
"""

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissIndex:
    """FAISS index for face embedding search (cosine via normalized inner product)."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.user_map = []  # index position → user name

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
            self._embeddings = []

    def build_from_users(self, users: dict):
        """Build index from {name: [[emb1], [emb2], ...]} dict."""
        self.user_map = []
        all_embs = []

        for name, embeddings in users.items():
            for emb in embeddings:
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                all_embs.append(vec)
                self.user_map.append(name)

        if not all_embs:
            return

        matrix = np.vstack(all_embs).astype(np.float32)

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(matrix)
        else:
            self._embeddings = [e.copy() for e in all_embs]

    def search(self, query_embedding: np.ndarray, k: int = 1) -> list:
        """Search for k nearest neighbors. Returns [{name, score, index}]."""
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query, min(k, self.index.ntotal))
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(self.user_map):
                    results.append({
                        "name": self.user_map[idx],
                        "score": float(scores[0][i]),
                        "index": int(idx),
                    })
            return results

        elif not FAISS_AVAILABLE and self._embeddings:
            scored = []
            for i, emb in enumerate(self._embeddings):
                s = float(np.dot(query.flatten(), np.array(emb, dtype=np.float32)))
                scored.append((i, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                {"name": self.user_map[idx], "score": sc, "index": idx}
                for idx, sc in scored[:k]
                if idx < len(self.user_map)
            ]

        return []

    def add_embedding(self, name: str, embedding: np.ndarray):
        """Add a single embedding to the live index."""
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vec)
        else:
            self._embeddings.append(vec.flatten())
        self.user_map.append(name)

    @property
    def total(self) -> int:
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return len(self._embeddings)
