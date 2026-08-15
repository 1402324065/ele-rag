from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import create_openai_client

router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(default=3, description="Reference document count")
    use_llm: Optional[bool] = Field(default=True, description="Whether to use LLM")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="System answer")
    sources: List[dict] = Field(default_factory=list, description="Reference documents")


def generate_rag_answer(
    query: str,
    sources: List[dict],
    settings,
    use_llm: bool = True,
) -> str:
    source_texts = [s.get("text", "") for s in sources]
    combined = "\n\n".join(source_texts)

    if not combined:
        return "No directly relevant content was found in the knowledge base."

    if not use_llm or not settings.openai_api_key:
        return "Retrieved knowledge base content:\n\n" + combined

    try:
        client = create_openai_client(settings, timeout=30.0)
        context_text = "\n\n".join(
            [f"Snippet {i + 1}:\n{s.get('text', '')}" for i, s in enumerate(sources)]
        )
        system_prompt = """
You are a professional power industry assistant.
Answer the user's question in clear Chinese using only the provided knowledge base snippets.
If the snippets do not contain the answer, say so directly.
Do not invent any information outside the snippets.
        """.strip()

        user_prompt = f"""User question: {query}

Knowledge base snippets:
{context_text}
        """.strip()

        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"LLM call failed: {exc}\n\nRetrieved knowledge base content:\n\n" + combined


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest):
    vector_store = request.app.state.vector_store
    index = request.app.state.index
    metadata = request.app.state.metadata
    backend = request.app.state.backend
    vectorizer = request.app.state.vectorizer
    settings = request.app.state.settings

    top_k = body.top_k if body.top_k else settings.top_k

    sources = vector_store.search(
        query=body.query,
        index=index,
        metadata=metadata,
        backend=backend,
        top_k=top_k,
        vectorizer=vectorizer,
    )

    answer = generate_rag_answer(body.query, sources, settings, use_llm=body.use_llm)
    return ChatResponse(answer=answer, sources=sources)


@router.get("/chat/test")
async def chat_test():
    return {
        "test": "ok",
        "example_query": "Transformer inspection points",
    }
