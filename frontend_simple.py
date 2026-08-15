import os
import json
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="??????????",
    page_icon="?",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .chat-message { padding: 1.2rem; border-radius: 0.8rem; margin-bottom: 1rem; }
    .user-message { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    .assistant-message { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; }
    h1, h2, h3 { color: #e0e7ff !important; }
</style>
""", unsafe_allow_html=True)

# Load resources WITHOUT st.cache_resource!
from app.core.config import get_settings
from app.rag.embedder import create_vector_store_from_index

settings = get_settings()
vector_store, index, metadata, backend, vectorizer = create_vector_store_from_index(
    settings.vector_store_dir, settings.embedding_model
)

llm_client = None
llm_available = False
if settings.openai_api_key and settings.openai_api_key != "your_api_key_here":
    try:
        llm_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        llm_available = True
    except:
        llm_available = False

st.sidebar.title("? ??????????")
st.sidebar.markdown("---")
st.sidebar.subheader("? ??")
st.sidebar.info(f"???????: {len(metadata)} ??")
st.sidebar.info(f"???????: {backend.upper()}")
if llm_available:
    st.sidebar.success("?????: ????")
else:
    st.sidebar.warning("?????: ??????")

use_llm = st.sidebar.checkbox("??????????", value=llm_available, disabled=not llm_available)

st.title("? ??????????")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "???????????????????????????????????", "sources": []}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-message user-message'><strong>? ???</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message assistant-message'><strong>? ????</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("? ???????", expanded=False):
                for i, s in enumerate(msg["sources"], 1):
                    st.write(f"**??? {i}** (???: {os.path.basename(s.get('source', 'unknown'))})")
                    st.write(s.get("text", ""))

st.markdown("---")
user_input = st.text_input("?????????????...", placeholder="????????????????????????")

if st.button("????????", type="primary", use_container_width=True):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
        
        with st.spinner("???????????..."):
            sources = vector_store.search(
                query=user_input, index=index, metadata=metadata, 
                backend=backend, top_k=3, vectorizer=vectorizer
            )
        
        with st.spinner("??????????..."):
            if use_llm and llm_available:
                context_text = "\n\n".join([f"??? {i+1}:\n{s.get('text', '')}" for i, s in enumerate(sources)])
                system_prompt = "???????????????????????????????????????????"
                user_prompt = f"????: {user_input}\n\n????:\n{context_text}"
                try:
                    response = llm_client.chat.completions.create(
                        model=settings.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=512
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = f"???????????: {e}\n\n????????:\n" + "\n\n".join([s.get("text", "") for s in sources])
            else:
                answer = "????????:\n" + "\n\n".join([s.get("text", "") for s in sources])
        
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
