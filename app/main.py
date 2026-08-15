from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.api import chat
from app.core.config import get_settings
from app.rag.embedder import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    vector_store = VectorStore(settings.embedding_model)
    index, metadata, backend, vectorizer = VectorStore.load(settings.vector_store_dir)
    app.state.vector_store = vector_store
    app.state.index = index
    app.state.metadata = metadata
    app.state.backend = backend
    app.state.vectorizer = vectorizer
    app.state.settings = settings
    yield


app = FastAPI(title="Power Knowledge Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])


@app.get("/")
async def root():
    return {"message": "Power Knowledge Assistant API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
