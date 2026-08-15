import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "\u7535\u529b\u77e5\u8bc6\u52a9\u624b"
APP_SUBTITLE = "\u57fa\u4e8e\u4f01\u4e1a\u77e5\u8bc6\u5e93\u7684\u667a\u80fd\u95ee\u7b54\u7cfb\u7edf"
WELCOME_TEXT = "\u60a8\u597d\uff01\u6211\u662f\u7535\u529b\u77e5\u8bc6\u52a9\u624b\u3002\u6709\u4ec0\u4e48\u53ef\u4ee5\u5e2e\u52a9\u60a8\u7684\uff1f"
NO_RESULT_TEXT = "\u62b1\u6b49\uff0c\u77e5\u8bc6\u5e93\u4e2d\u6682\u672a\u627e\u5230\u4e0e\u8be5\u95ee\u9898\u76f4\u63a5\u76f8\u5173\u7684\u5185\u5bb9\u3002"
RETRIEVED_PREFIX = "\u77e5\u8bc6\u5e93\u4e2d\u627e\u5230\u7684\u76f8\u5173\u5185\u5bb9\uff1a\n\n"
LLM_FAIL_PREFIX = "\u5927\u8bed\u8a00\u6a21\u578b\u8c03\u7528\u5931\u8d25\uff1a"


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        .main .block-container { padding-top: 2rem; }
        .chat-message {
            padding: 1.2rem;
            border-radius: 0.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .user-message { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
        .assistant-message { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; }
        .source-box {
            background-color: rgba(0,0,0,0.3);
            border-left: 4px solid #f59e0b;
            padding: 0.8rem;
            margin-top: 0.5rem;
            border-radius: 0 0.5rem 0.5rem 0;
            font-size: 0.9rem;
            color: #d1d5db;
        }
        h1, h2, h3 { color: #e0e7ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="\u6b63\u5728\u52a0\u8f7d\u77e5\u8bc6\u5e93\u548c\u6a21\u578b...")
def load_resources():
    from app.core.config import create_openai_client, get_settings
    from app.rag.embedder import create_vector_store_from_index

    settings = get_settings()
    vector_store, index, metadata, backend, vectorizer = create_vector_store_from_index(
        settings.vector_store_dir,
        settings.embedding_model,
    )

    llm_client = None
    llm_available = False
    if settings.openai_api_key and settings.openai_api_key != "your_api_key_here":
        try:
            llm_client = create_openai_client(settings, timeout=30.0)
            llm_available = True
        except Exception as exc:
            st.warning(f"\u5927\u8bed\u8a00\u6a21\u578b\u5ba2\u6237\u7aef\u521d\u59cb\u5316\u5931\u8d25: {exc}")

    return vector_store, index, metadata, backend, vectorizer, settings, llm_client, llm_available


def build_answer(user_query, sources, use_llm, llm_available, llm_client, settings):
    if not sources:
        return NO_RESULT_TEXT

    combined = "\n\n".join(source.get("text", "") for source in sources)
    if not use_llm or not llm_available or llm_client is None:
        return RETRIEVED_PREFIX + combined

    context_text = "\n\n".join(
        [f"Snippet {idx + 1}:\n{source.get('text', '')}" for idx, source in enumerate(sources)]
    )
    system_prompt = (
        "You are a professional power industry assistant. "
        "Answer the user's question in clear Chinese using only the provided knowledge base snippets. "
        "If the snippets do not contain the answer, say so directly. "
        "Do not invent any information outside the snippets."
    )
    user_prompt = f"User question: {user_query}\n\nKnowledge base snippets:\n{context_text}"

    try:
        response = llm_client.chat.completions.create(
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
        return f"{LLM_FAIL_PREFIX} {exc}\n\n{RETRIEVED_PREFIX}{combined}"


def render_message(role, content):
    label = "\u7528\u6237" if role == "user" else "\u52a9\u624b"
    css_class = "user-message" if role == "user" else "assistant-message"
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f'<div class="chat-message {css_class}"><strong>{label}</strong><br>{safe_content}</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources):
    with st.expander("\u53c2\u8003\u8d44\u6599", expanded=False):
        for idx, source in enumerate(sources, start=1):
            score = source.get("score", "N/A")
            file_name = os.path.basename(source.get("source", "unknown"))
            text = html.escape(source.get("text", "")).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="source-box">
                    <strong>\u7247\u6bb5 {idx} (\u76f8\u4f3c\u5ea6: {score})</strong><br>
                    <small>\u6765\u6e90: {html.escape(file_name)}</small><br>
                    <p style="margin-top: 0.5rem;">{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


(
    vector_store,
    index,
    metadata,
    backend,
    vectorizer,
    settings,
    llm_client,
    llm_available,
) = load_resources()

st.sidebar.title(APP_TITLE)
st.sidebar.markdown("---")
st.sidebar.subheader("\u8bbe\u7f6e")
top_k = st.sidebar.slider("\u53c2\u8003\u6587\u6863\u6570\u91cf (Top K)", min_value=1, max_value=5, value=settings.top_k)
use_llm = st.sidebar.checkbox(
    "\u542f\u7528\u5927\u8bed\u8a00\u6a21\u578b\u667a\u80fd\u56de\u7b54",
    value=llm_available,
    disabled=not llm_available,
)

if not llm_available:
    st.sidebar.info("\u63d0\u793a: \u7f16\u8f91 `.env` \u6587\u4ef6\u4ee5\u914d\u7f6e\u5927\u8bed\u8a00\u6a21\u578b\u3002")

st.sidebar.markdown("---")
st.sidebar.subheader("\u72b6\u6001")
st.sidebar.info(f"\u77e5\u8bc6\u5e93\u6587\u6863: {len(metadata)} \u4e2a\u7247\u6bb5")
st.sidebar.info(f"\u68c0\u7d22\u540e\u7aef: {backend.upper()}")

if llm_available:
    st.sidebar.success("\u5927\u8bed\u8a00\u6a21\u578b\u8fde\u63a5: \u6b63\u5e38")
else:
    st.sidebar.warning("\u5927\u8bed\u8a00\u6a21\u578b\u8fde\u63a5: \u672a\u914d\u7f6e")

st.sidebar.markdown("---")
st.sidebar.subheader("\u793a\u4f8b\u95ee\u9898")
example_questions = [
    "\u53d8\u538b\u5668\u5de1\u68c0\u8981\u70b9\u6709\u54ea\u4e9b\uff1f",
    "\u5f00\u5173\u67dc\u5de1\u68c0\u8981\u70b9\u6709\u54ea\u4e9b\uff1f",
    "\u53d8\u538b\u5668\u8fc7\u6e29\u600e\u4e48\u529e\uff1f",
    "\u65ad\u8def\u5668\u8df3\u95f8\u600e\u4e48\u5904\u7406\uff1f",
    "\u5de1\u68c0\u8bb0\u5f55\u6709\u4ec0\u4e48\u8981\u6c42\uff1f",
]
for idx, question in enumerate(example_questions):
    if st.sidebar.button(question, use_container_width=True, key=f"example_btn_{idx}"):
        st.session_state["user_input"] = question

st.title(APP_TITLE)
st.markdown(APP_SUBTITLE)
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_TEXT, "sources": []}]

for message in st.session_state.messages:
    render_message(message["role"], message["content"])
    if message["role"] == "assistant" and message.get("sources"):
        render_sources(message["sources"])

st.markdown("---")
user_input = st.text_input(
    "\u8bf7\u8f93\u5165\u60a8\u7684\u95ee\u9898...",
    placeholder="\u4f8b\u5982: \u53d8\u538b\u5668\u5de1\u68c0\u8981\u70b9\u6709\u54ea\u4e9b\uff1f",
    key="user_input",
)

if st.button("\u53d1\u9001\u95ee\u9898", type="primary", use_container_width=True, key="send_btn"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})

        with st.spinner("\u6b63\u5728\u641c\u7d22\u77e5\u8bc6\u5e93..."):
            sources = vector_store.search(
                query=user_input,
                index=index,
                metadata=metadata,
                backend=backend,
                top_k=top_k,
                vectorizer=vectorizer,
            )

        with st.spinner("\u6b63\u5728\u751f\u6210\u56de\u7b54..."):
            answer = build_answer(
                user_query=user_input,
                sources=sources,
                use_llm=use_llm,
                llm_available=llm_available,
                llm_client=llm_client,
                settings=settings,
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()
