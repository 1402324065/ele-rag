from __future__ import annotations

from dataclasses import asdict, dataclass

from app.rag.document_loader import LoadedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    step = chunk_size - chunk_overlap

    for doc_index, document in enumerate(documents):
        text = document.text.strip()
        if not text:
            continue

        for chunk_index, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start : start + chunk_size].strip()
            if not chunk_text:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=f"doc-{doc_index}-chunk-{chunk_index}",
                    source=document.source,
                    text=chunk_text,
                )
            )

    return chunks
