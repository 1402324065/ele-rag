from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.rag.text_chunker import DocumentChunk


class VectorStore:
    def __init__(self, model_name: str, backend: str = "tfidf") -> None:
        self.model_name = model_name
        self.backend = backend
        self.model: Any | None = None
        self.vectorizer: TfidfVectorizer | None = None

        if self.backend == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
            except Exception as exc:
                self.backend = "tfidf"
                warnings.warn(
                    (
                        f"Failed to load embedding model '{model_name}', "
                        "fallback to local TF-IDF retrieval. "
                        f"Original error: {exc}"
                    ),
                    stacklevel=2,
                )

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SentenceTransformer backend is not available.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def build_index(self, chunks: list[DocumentChunk]) -> tuple[Any, list[dict[str, str]]]:
        if not chunks:
            raise ValueError("No chunks available to build the vector index.")

        texts = [chunk.text for chunk in chunks]
        if self.backend == "sentence_transformer":
            embeddings = self.encode(texts)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                lowercase=False,
                sublinear_tf=True,
            )
            index = self.vectorizer.fit_transform(texts)

        metadata = [chunk.to_dict() for chunk in chunks]
        return index, metadata

    @staticmethod
    def save(
        index: Any,
        metadata: list[dict[str, str]],
        vector_store_dir: Path,
        backend: str,
        vectorizer: TfidfVectorizer | None = None,
    ) -> None:
        vector_store_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"backend": backend}
        (vector_store_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (vector_store_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if backend == "sentence_transformer":
            faiss.write_index(index, str(vector_store_dir / "index.faiss"))
            return

        if vectorizer is None:
            raise ValueError("TF-IDF backend requires a fitted vectorizer.")

        sparse.save_npz(str(vector_store_dir / "tfidf_index.npz"), index)
        with (vector_store_dir / "tfidf_vectorizer.pkl").open("wb") as file_obj:
            pickle.dump(vectorizer, file_obj)

    @staticmethod
    def load(vector_store_dir: Path) -> tuple[Any, list[dict[str, str]], str, TfidfVectorizer | None]:
        manifest_path = vector_store_dir / "manifest.json"
        metadata_path = vector_store_dir / "metadata.json"

        if not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Vector index files not found. Please run the build script first.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend = manifest.get("backend", "tfidf")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if backend == "sentence_transformer":
            index_path = vector_store_dir / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError("FAISS index file not found.")
            index = faiss.read_index(str(index_path))
            return index, metadata, backend, None

        index_path = vector_store_dir / "tfidf_index.npz"
        vectorizer_path = vector_store_dir / "tfidf_vectorizer.pkl"
        if not index_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("TF-IDF index files not found.")

        index = sparse.load_npz(str(index_path))
        with vectorizer_path.open("rb") as file_obj:
            vectorizer = pickle.load(file_obj)
        return index, metadata, backend, vectorizer

    def search(
        self,
        query: str,
        index: Any,
        metadata: list[dict[str, str]],
        backend: str,
        top_k: int = 3,
        vectorizer: TfidfVectorizer | None = None,
    ) -> list[dict[str, str]]:
        if backend == "sentence_transformer":
            query_embedding = self.encode([query])
            scores, indices = index.search(query_embedding, top_k)

            results: list[dict[str, str]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                item = dict(metadata[idx])
                item["score"] = f"{float(score):.4f}"
                results.append(item)
            return results

        if vectorizer is None:
            raise ValueError("TF-IDF search requires a fitted vectorizer.")

        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, index)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[dict[str, str]] = []
        for idx in top_indices:
            item = dict(metadata[idx])
            item["score"] = f"{float(similarities[idx]):.4f}"
            results.append(item)
        return results


# Helper function to create VectorStore from existing index
def create_vector_store_from_index(
    vector_store_dir: Path,
    embedding_model_name: str
) -> tuple[VectorStore, Any, list[dict[str, str]], str, Any]:
    index, metadata, backend, vectorizer = VectorStore.load(vector_store_dir)
    vector_store = VectorStore(embedding_model_name, backend=backend)
    vector_store.vectorizer = vectorizer
    return vector_store, index, metadata, backend, vectorizer
