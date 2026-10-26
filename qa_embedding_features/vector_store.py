"""Persistent ChromaDB adapter for the project's precomputed embeddings."""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from pathlib import Path


class ChromaVectorStore:
    """Store and retrieve explicit vectors in a persistent local Chroma collection."""

    def __init__(self, path: Path, model: str | None = None, dimensions: int | None = None,
                 cache_size: int = 512):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "No está instalado ChromaDB. Ejecuta: python -m pip install -r requirements.txt"
            ) from exc
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.model = model
        self.dimensions = dimensions
        self.cache_size = cache_size
        self.cache: OrderedDict[str, list[float]] = OrderedDict()
        self._collections = {}
        self.collection = self._collection(model, dimensions) if model and dimensions else None

    @staticmethod
    def _collection_name(model: str, dimensions: int) -> str:
        safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", model).strip("_-").lower() or "model"
        name = f"qa_{safe_model}_{dimensions}d"
        return name[:63].rstrip("_-.")

    @staticmethod
    def _digest(value: str, length: int = 24) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]

    def _collection(self, model: str, dimensions: int):
        key = (model, dimensions)
        if key not in self._collections:
            name = self._collection_name(model, dimensions)
            self._collections[key] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "embedding_model": model,
                          "embedding_dimensions": dimensions},
            )
        return self._collections[key]

    def _all_collections(self):
        collections = []
        for item in self.client.list_collections():
            name = item if isinstance(item, str) else item.name
            collections.append(item if not isinstance(item, str) else self.client.get_collection(name=name))
        return collections

    def text_id(self, text: str) -> str:
        if not self.model or not self.dimensions:
            raise ValueError("El almacén requiere modelo y dimensiones para generar IDs de texto.")
        return "E-" + self._digest(f"{self.model}:{self.dimensions}:{text}")

    def centroid_id(self, context_id: str, paragraph_ids: list[str]) -> str:
        material = f"{self.model}:{self.dimensions}:{context_id}:" + ":".join(paragraph_ids)
        return "EC-" + self._digest(material)

    def _remember(self, embedding_id: str, vector: list[float]) -> None:
        self.cache[embedding_id] = vector
        self.cache.move_to_end(embedding_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def get(self, embedding_id: str):
        cached = self.cache.get(embedding_id)
        if cached is not None:
            self.cache.move_to_end(embedding_id)
            return cached
        collections = [self.collection] if self.collection is not None else self._all_collections()
        for collection in collections:
            result = collection.get(ids=[embedding_id], include=["embeddings"])
            vectors = result.get("embeddings")
            if vectors is not None and len(vectors):
                vector = vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])
                self._remember(embedding_id, vector)
                return vector
        return None

    def get_many(self, embedding_ids: list[str], batch_size: int = 256) -> dict[str, list[float]]:
        requested = list(dict.fromkeys(embedding_ids))
        found = {embedding_id: self.cache[embedding_id] for embedding_id in requested
                 if embedding_id in self.cache}
        missing = [embedding_id for embedding_id in requested if embedding_id not in found]
        collections = [self.collection] if self.collection is not None else self._all_collections()
        for collection in collections:
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                result = collection.get(ids=batch, include=["embeddings"])
                vectors = result.get("embeddings")
                if vectors is None:
                    continue
                for embedding_id, raw_vector in zip(result.get("ids", []), vectors):
                    vector = raw_vector.tolist() if hasattr(raw_vector, "tolist") else list(raw_vector)
                    found[embedding_id] = vector
                    self._remember(embedding_id, vector)
            missing = [embedding_id for embedding_id in missing if embedding_id not in found]
            if not missing:
                break
        return found

    def existing_ids(self, embedding_ids: list[str], batch_size: int = 512) -> set[str]:
        existing: set[str] = set()
        if self.collection is None:
            return existing
        unique_ids = list(dict.fromkeys(embedding_ids))
        for start in range(0, len(unique_ids), batch_size):
            batch = unique_ids[start:start + batch_size]
            result = self.collection.get(ids=batch, include=["metadatas"])
            existing.update(result.get("ids", []))
        return existing

    def get_record(self, embedding_id: str):
        collections = [self.collection] if self.collection is not None else self._all_collections()
        for collection in collections:
            result = collection.get(ids=[embedding_id], include=["embeddings", "metadatas", "documents"])
            vectors = result.get("embeddings")
            if vectors is None or not len(vectors):
                continue
            vector = vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])
            metadata = (result.get("metadatas") or [{}])[0] or {}
            document = (result.get("documents") or [None])[0]
            self._remember(embedding_id, vector)
            return {"embedding_id": embedding_id, "model": metadata.get("model"),
                    "dimensions": int(metadata.get("dimensions", len(vector))),
                    "kind": metadata.get("kind", "text"), "document": document,
                    "vector": vector}
        return None

    def put(self, embedding_id: str, vector: list[float], kind: str,
            document: str | None = None) -> None:
        if self.collection is None or not self.model or not self.dimensions:
            raise ValueError("El almacén fue abierto en modo lectura; indica modelo y dimensiones para escribir.")
        if len(vector) != self.dimensions:
            raise ValueError(f"El vector recibido tiene {len(vector)} dimensiones; se esperaban {self.dimensions}.")
        metadata = {"kind": kind, "model": self.model, "dimensions": self.dimensions}
        args = {"ids": [embedding_id], "embeddings": [vector], "metadatas": [metadata]}
        if document:
            args["documents"] = [document]
        self.collection.upsert(**args)
        self._remember(embedding_id, vector)

    def put_many(self, embedding_ids: list[str], vectors: list[list[float]], kind: str,
                 documents: list[str] | None = None) -> None:
        """Persist a batch of vectors in one ChromaDB write."""
        if self.collection is None or not self.model or not self.dimensions:
            raise ValueError("El almacén fue abierto en modo lectura; indica modelo y dimensiones para escribir.")
        if len(embedding_ids) != len(vectors):
            raise ValueError("La cantidad de IDs y vectores debe coincidir.")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(f"Todos los vectores deben tener {self.dimensions} dimensiones.")
        metadatas = [{"kind": kind, "model": self.model,
                      "dimensions": self.dimensions} for _ in embedding_ids]
        args = {"ids": embedding_ids, "embeddings": vectors, "metadatas": metadatas}
        if documents is not None:
            args["documents"] = documents
        self.collection.upsert(**args)
        for embedding_id, vector in zip(embedding_ids, vectors):
            self._remember(embedding_id, vector)

    def commit(self) -> None:
        """Compatibility method; PersistentClient writes through to disk."""
        return None

    def close(self) -> None:
        self.cache.clear()
