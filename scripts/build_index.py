from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.rag.document_loader import load_documents
from app.rag.embedder import VectorStore
from app.rag.text_chunker import chunk_documents


def main() -> None:
    settings = get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(settings.data_dir)
    if not documents:
        print(f"No supported documents found in: {settings.data_dir}")
        print("Please add .txt, .pdf or .docx files into data/raw and rerun the script.")
        return

    chunks = chunk_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Keep a readable copy of the cleaned text for later inspection.
    for document in documents:
        source_name = Path(document.source).stem
        output_path = settings.processed_dir / f"{source_name}.txt"
        output_path.write_text(document.text, encoding="utf-8")

    vector_store = VectorStore(settings.embedding_model, backend="tfidf")
    index, metadata = vector_store.build_index(chunks)
    vector_store.save(
        index=index,
        metadata=metadata,
        vector_store_dir=settings.vector_store_dir,
        backend=vector_store.backend,
        vectorizer=vector_store.vectorizer,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Retrieval backend: {vector_store.backend}")
    print(f"Saved vector index to: {settings.vector_store_dir}")


if __name__ == "__main__":
    main()
