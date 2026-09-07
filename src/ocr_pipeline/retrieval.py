from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from .chunking import Chunk


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class QdrantIndex:
    def __init__(self, path: Path = Path("data/qdrant"), collection: str = "ocr_chunks", embedder: LocalEmbedder | None = None, url: str | None = None) -> None:
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url, path=None if url else str(path))
        self.collection = collection
        self.embedder = embedder or LocalEmbedder()

    def add(self, chunks: Iterable[Chunk], batch_size: int = 32) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        items = list(chunks)
        if not items:
            return
        embeddings = self.embedder.embed([chunk.text for chunk in items], batch_size)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
            )
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector=embedding,
                    payload={"text": chunk.text, **chunk.metadata()},
                )
                for chunk, embedding in zip(items, embeddings)
            ],
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        embedding = self.embedder.embed([query], 1)[0]
        result = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=k,
            with_payload=True,
        )
        return [
            {"text": point.payload.get("text", ""), "metadata": {key: value for key, value in point.payload.items() if key != "text"}, "distance": point.score}
            for point in result.points
        ]
