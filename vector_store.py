# ========= vector_store.py (persistencia en ./biorag_store) 

import os, time, hashlib
from typing import List, Dict, Any
import numpy as np
import pandas as pd

def _sha256_norm(text: str) -> str:
    norm = " ".join((text or "").split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class VectorStore:
    """
    Guarda embeddings en ./biorag_store/embeddings.npy y metadatos en ./biorag_store/meta.parquet.
    """
    def __init__(self, path: str, embed_fn):
        self.path = path
        self.embed_fn = embed_fn
        os.makedirs(self.path, exist_ok=True)
        self._emb_path = os.path.join(self.path, "embeddings.npy")
        self._meta_path = os.path.join(self.path, "meta.parquet")
        self._load()

    def _load(self):
        if os.path.exists(self._emb_path):
            self.emb = np.load(self._emb_path)
        else:
            self.emb = np.zeros((0, 0), dtype=np.float32)
        if os.path.exists(self._meta_path):
            self.meta = pd.read_parquet(self._meta_path)
        else:
            self.meta = pd.DataFrame(columns=[
                "row_id","doc_id","chunk_id","hash","source","title","model","dim","created_at"
            ])
        self.dim = None if self.emb.size == 0 else self.emb.shape[1]

    def _persist(self):
        np.save(self._emb_path, self.emb.astype(np.float32))
        self.meta.to_parquet(self._meta_path, index=False)

    def nuevos(self, chunks: List[Dict[str, Any]], model_name: str, batch_size: int = 64) -> int:
        """
        chunks: [{"doc_id": str, "chunk_id": int, "text": str, "source": str, "title": str}, ...]
        Devuelve cuántos embeddings se agregaron.
        """
        if not chunks:
            return 0

        existing = set(self.meta["hash"]) if len(self.meta) else set()
        new_chunks = []
        for c in chunks:
            h = _sha256_norm(c.get("text",""))
            if h not in existing:
                cc = dict(c)
                cc["hash"] = h
                new_chunks.append(cc)

        if not new_chunks:
            return 0

        texts = [c["text"] for c in new_chunks]
        embs = []
        for i in range(0, len(texts), batch_size):
            e = self.embed_fn(texts[i:i+batch_size]).astype(np.float32)
            embs.append(e)
        embs = np.vstack(embs) if embs else np.zeros((0, self.dim or 0), dtype=np.float32)

        if self.dim is None and embs.size > 0:
            self.dim = embs.shape[1]

        elif embs.size > 0 and embs.shape[1] != self.dim:
            raise ValueError(
                f"Dimensión inconsistente: store={self.dim}, intentás agregar={embs.shape[1]}. "
                f"Usá otra carpeta o limpiá el store si cambiaste de modelo de embeddings."
                )

        if self.emb.size == 0:
            self.emb = embs
        else:
            self.emb = np.vstack([self.emb, embs])

        start_id = 0 if len(self.meta)==0 else int(self.meta["row_id"].max())+1
        now = int(time.time())
        rows = []
        for i, c in enumerate(new_chunks):
            rows.append({
                "row_id": start_id + i,
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id", 0),
                "hash": c["hash"],
                "source": c.get("source"),
                "title": c.get("title"),
                "model": model_name,
                "dim": self.dim,
                "created_at": now,
            })
        self.meta = pd.concat([self.meta, pd.DataFrame(rows)], ignore_index=True)

        self._persist()
        return len(new_chunks)

    def embeddings_x_hash(self, hashes: List[str]) -> np.ndarray:
        """
        Devuelve una matriz de embeddings en el mismo orden de hashes.
        Asume que todos esos hashes existen en self.meta (nuevos antes si no).
        """
        if not hashes:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        meta_idx = self.meta.set_index("hash")
        row_ids = [int(meta_idx.loc[h]["row_id"]) for h in hashes]
        return self.emb[row_ids]