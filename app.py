import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
import asyncio
import re
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lexis · Intelligent Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  –  UI8 Dubai luxury dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #08080f !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(180,140,80,.13) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(100,80,180,.07) 0%, transparent 60%),
        #08080f !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"] { color: #a89060 !important; }
[data-testid="collapsedControl"] { color: #a89060 !important; }
[data-testid="collapsedControl"] { display: none !important; }
             
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
    width: 320px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
    background: rgba(12,11,20,.95) !important;
    border-right: 1px solid rgba(180,140,60,.14) !important;
    backdrop-filter: blur(20px);
}

/* hide collapse button completely */
[data-testid="collapsedControl"] {
    display: none !important;
}
                        
/* ── Wordmark ── */
.lexis-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: .04em;
    color: #d4aa60;
    display: flex;
    align-items: center;
    gap: .5rem;
    margin-bottom: .25rem;
    animation: fadeSlideDown .6s ease both;
}
.lexis-tagline {
    font-size: .72rem;
    font-weight: 300;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: rgba(200,185,150,.45);
    margin-bottom: 2rem;
    animation: fadeSlideDown .7s ease both;
}
.diamond { color: #d4aa60; font-size: 1.1rem; }

/* ── Section labels ── */
.sidebar-label {
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: rgba(200,185,140,.38);
    margin: 1.6rem 0 .6rem;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
.stTextInput input {
    background: rgba(255,255,255,.035) !important;
    border: 1px solid rgba(180,150,70,.2) !important;
    border-radius: 10px !important;
    color: #e0d8c8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .82rem !important;
    padding: .65rem 1rem !important;
    transition: border-color .25s, box-shadow .25s;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(212,170,96,.55) !important;
    box-shadow: 0 0 0 3px rgba(212,170,96,.08) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label,
.stTextInput label {
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,.5) !important;
    margin-bottom: .4rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(212,170,96,.22) !important;
    border-radius: 12px !important;
    background: rgba(212,170,96,.025) !important;
    padding: 1rem !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(212,170,96,.45) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(200,185,140,.6) !important;
    font-size: .78rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(212,170,96,.35) !important;
    color: #d4aa60 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: .55rem 1.2rem !important;
    transition: all .22s ease !important;
}
.stButton > button:hover {
    background: rgba(212,170,96,.1) !important;
    border-color: rgba(212,170,96,.7) !important;
    box-shadow: 0 0 18px rgba(212,170,96,.12) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Process primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(212,170,96,.2), rgba(180,130,60,.35)) !important;
    border-color: rgba(212,170,96,.6) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    color: rgba(200,185,150,.6) !important;
    font-size: .78rem !important;
}
[data-testid="stRadio"] > div { gap: .5rem !important; }

/* ── Main header ── */
.main-header {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    animation: fadeSlideDown .8s ease both;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 600;
    color: #e8e0cc;
    line-height: 1.15;
    letter-spacing: -.01em;
}
.main-title em {
    font-style: italic;
    color: #d4aa60;
}
.main-subtitle {
    margin-top: .9rem;
    font-size: .85rem;
    font-weight: 300;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: rgba(200,185,140,.4);
}
.header-rule {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d4aa60, transparent);
    margin: 1.4rem auto 0;
}

/* ── Query input card ── */
.query-wrap {
    max-width: 820px;
    margin: 0 auto 2.5rem;
    animation: fadeSlideUp .7s .2s ease both;
}
.query-label {
    font-size: .7rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: rgba(200,185,140,.4);
    margin-bottom: .6rem;
}

/* ── Response card ── */
.response-card {
    max-width: 820px;
    margin: 0 auto 1.6rem;
    background: rgba(255,255,255,.028);
    border: 1px solid rgba(212,170,96,.12);
    border-radius: 16px;
    overflow: hidden;
    animation: fadeSlideUp .5s ease both;
}
.response-header {
    display: flex;
    align-items: center;
    gap: .75rem;
    padding: .9rem 1.4rem;
    border-bottom: 1px solid rgba(212,170,96,.08);
    background: rgba(212,170,96,.04);
}
.response-role {
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: .18em;
    text-transform: uppercase;
}
.role-user { color: rgba(180,200,255,.6); }
.role-assistant { color: #d4aa60; }
.role-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}
.response-body {
    padding: 1.3rem 1.6rem;
    font-size: .92rem;
    line-height: 1.75;
    color: #ddd5c0;
}

/* ── Keyword suggestions ── */
.keywords-wrap {
    max-width: 820px;
    margin: 0 auto 2.5rem;
    animation: fadeSlideUp .6s .1s ease both;
}
.keywords-title {
    font-size: .68rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: rgba(200,185,140,.35);
    margin-bottom: .75rem;
}
.keywords-grid {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
}
.keyword-chip {
    display: inline-block;
    padding: .42rem 1rem;
    border: 1px solid rgba(212,170,96,.22);
    border-radius: 100px;
    font-size: .76rem;
    color: rgba(212,170,96,.8);
    background: rgba(212,170,96,.04);
    cursor: pointer;
    transition: all .2s;
    font-family: 'DM Sans', sans-serif;
}
.keyword-chip:hover {
    background: rgba(212,170,96,.12);
    border-color: rgba(212,170,96,.55);
    transform: translateY(-1px);
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 2rem;
    justify-content: center;
    margin: 0 auto 3rem;
    max-width: 820px;
    animation: fadeSlideUp .7s .3s ease both;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #d4aa60;
    display: block;
}
.stat-label {
    font-size: .65rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: rgba(200,185,140,.35);
}

/* ── Empty state ── */
.empty-state {
    max-width: 480px;
    margin: 3rem auto;
    text-align: center;
    opacity: .55;
    animation: pulse 3s ease-in-out infinite;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1.2rem;
    display: block;
}
.empty-text {
    font-size: .85rem;
    line-height: 1.6;
    color: rgba(200,185,140,.5);
}

/* ── Download link ── */
.download-btn {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    padding: .55rem 1.2rem;
    border: 1px solid rgba(212,170,96,.3);
    border-radius: 8px;
    color: #d4aa60 !important;
    text-decoration: none !important;
    font-size: .76rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    background: transparent;
    transition: all .22s;
}
.download-btn:hover {
    background: rgba(212,170,96,.1);
    border-color: rgba(212,170,96,.6);
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #d4aa60 !important; }

/* ── Success / Warning ── */
.stSuccess, .stWarning { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(212,170,96,.2); border-radius: 2px; }

/* ── Divider ── */
hr { border-color: rgba(212,170,96,.1) !important; }

/* ── Keyframes ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: .45; }
    50%       { opacity: .65; }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
.shimmer {
    background: linear-gradient(90deg, rgba(212,170,96,.1) 25%, rgba(212,170,96,.3) 50%, rgba(212,170,96,.1) 75%);
    background-size: 200% auto;
    animation: shimmer 2s linear infinite;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_qa_chain(api_key):
    prompt_template = """
You are Lexis, a precise and eloquent document intelligence assistant.
Using only the context provided, deliver a thorough, well-structured answer.
If the information is not present in the context, respond with:
"This detail is not available within the uploaded documents."
Never fabricate information.

Context:
{context}

Question:
{question}

Provide a clear, detailed, and professionally worded answer:
"""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_keyword_suggestions(api_key, user_question, answer_text):
    """Ask Gemini to suggest 6 follow-up keywords/short questions."""
    suggestion_prompt = f"""
Based on the following question and answer from a document, suggest exactly 6 short, insightful follow-up questions or keyword phrases (max 7 words each) that a reader might want to explore next.

Original question: {user_question}
Answer summary: {answer_text[:600]}

Return ONLY a JSON array of 6 strings. Example:
["What are the key financial metrics?", "Regulatory compliance overview", ...]
Do not include any other text or explanation.
"""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5,
            google_api_key=api_key,
        )
        response = model.invoke(suggestion_prompt)
        raw = response.content.strip()
        # strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        import json
        suggestions = json.loads(raw)
        return suggestions[:6] if isinstance(suggestions, list) else []
    except Exception:
        return []


def render_message(role: str, content: str):
    role_label = "You" if role == "user" else "◈ Lexis"
    role_class = "role-user" if role == "user" else "role-assistant"
    st.markdown(f"""
    <div class="response-card">
        <div class="response-header">
            <span class="role-dot {role_class}"></span>
            <span class="response-role {role_class}">{role_label}</span>
        </div>
        <div class="response-body">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_keywords(keywords: list):
    if not keywords:
        return
    chips = "".join(
        f'<span class="keyword-chip" onclick="void(0)">{kw}</span>'
        for kw in keywords
    )
    st.markdown(f"""
    <div class="keywords-wrap">
        <div class="keywords-title">◦ Suggested Explorations</div>
        <div class="keywords-grid">{chips}</div>
    </div>
    """, unsafe_allow_html=True)

    # Render clickable buttons below (Streamlit-native for interactivity)
    cols = st.columns(min(3, len(keywords)))
    for i, kw in enumerate(keywords[:6]):
        with cols[i % 3]:
            if st.button(kw, key=f"kw_{i}_{kw[:10]}"):
                st.session_state.prefill_question = kw
                st.rerun()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("conversation_history", []),
    ("last_keywords", []),
    ("prefill_question", ""),
    ("processed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="lexis-wordmark"><span class="diamond">◈</span> Lexis</div>', unsafe_allow_html=True)
    st.markdown('<div class="lexis-tagline">Document Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">AI Model</div>', unsafe_allow_html=True)
    model_name = st.radio("", ("Google Gemini",), label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">API Credentials</div>', unsafe_allow_html=True)
    api_key = st.text_input("Google AI API Key", type="password", placeholder="AIza••••••••••••••••")
    if not api_key:
        st.warning("An API key is required to proceed.")

    st.markdown('<div class="sidebar-label">Document Corpus</div>', unsafe_allow_html=True)
    pdf_docs = st.file_uploader(
        "Upload PDF documents",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    process_clicked = col1.button("Analyse", type="primary")
    reset_clicked = col2.button("Clear Session")

    if process_clicked:
        if pdf_docs and api_key:
            with st.spinner("Indexing corpus…"):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.session_state.processed = True
            st.success(f"✦ {len(pdf_docs)} document(s) indexed — {len(chunks)} segments")
        elif not api_key:
            st.error("Please enter your API key.")
        else:
            st.error("Please upload at least one PDF.")

    if reset_clicked:
        st.session_state.conversation_history = []
        st.session_state.last_keywords = []
        st.session_state.prefill_question = ""
        st.session_state.processed = False
        st.rerun()

    # Download history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown('<div class="sidebar-label">Export</div>', unsafe_allow_html=True)
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Timestamp", "Documents"],
        )
        csv_bytes = base64.b64encode(df.to_csv(index=False).encode()).decode()
        st.markdown(
            f'<a class="download-btn" href="data:file/csv;base64,{csv_bytes}" download="lexis_session.csv">↓ Export Session Log</a>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">Document <em>Intelligence</em>,<br>Redefined</div>
    <div class="main-subtitle">Semantic analysis · Contextual retrieval · Precision answers</div>
    <div class="header-rule"></div>
</div>
""", unsafe_allow_html=True)

# Stats bar
doc_count = len(pdf_docs) if pdf_docs else 0
q_count = len(st.session_state.conversation_history)
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-num">{doc_count}</span>
        <span class="stat-label">Documents Loaded</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">{q_count}</span>
        <span class="stat-label">Queries This Session</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">{'Active' if st.session_state.processed else '—'}</span>
        <span class="stat-label">Index Status</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Query input ──
st.markdown('<div class="query-wrap"><div class="query-label">◦ Pose your enquiry</div></div>', unsafe_allow_html=True)

prefill = st.session_state.get("prefill_question", "")
user_question = st.text_input(
    "",
    value=prefill,
    placeholder="What are the key findings in this document?",
    label_visibility="collapsed",
)
# clear prefill after use
if prefill:
    st.session_state.prefill_question = ""

# ── Process query ──
if user_question and api_key:
    if not st.session_state.processed:
        st.warning("Please upload documents and click **Analyse** before submitting a query.")
    else:
        with st.spinner("Retrieving insights…"):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question, k=5)
            chain = get_qa_chain(api_key)
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True,
            )
            answer = response["output_text"]

        # Store in history
        pdf_names = [p.name for p in pdf_docs] if pdf_docs else []
        st.session_state.conversation_history.append((
            user_question,
            answer,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ", ".join(pdf_names),
        ))

        # Render current Q&A
        render_message("user", user_question)
        render_message("assistant", answer)

        # Keyword suggestions
        with st.spinner("Generating follow-up suggestions…"):
            keywords = get_keyword_suggestions(api_key, user_question, answer)
            st.session_state.last_keywords = keywords

elif user_question and not api_key:
    st.warning("Please enter your Google AI API key in the sidebar.")

# ── Keyword chips ──
if st.session_state.last_keywords:
    render_keywords(st.session_state.last_keywords)

# ── Previous exchanges ──
history = st.session_state.conversation_history
if len(history) > 1:
    st.markdown("---")
    st.markdown('<div style="font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;color:rgba(200,185,140,.3);text-align:center;margin-bottom:1.4rem;">◦ Prior Exchanges</div>', unsafe_allow_html=True)
    for q, a, ts, pdfs in reversed(history[:-1]):
        render_message("user", q)
        render_message("assistant", a)

# ── Empty state ──
if not st.session_state.conversation_history and not user_question:
    st.markdown("""
    <div class="empty-state">
        <span class="empty-icon">◈</span>
        <div class="empty-text">
            Upload your documents via the sidebar,<br>
            click <strong>Analyse</strong>, then pose any question<br>
            to unlock the intelligence within your corpus.
        </div>
    </div>
    """, unsafe_allow_html=True)