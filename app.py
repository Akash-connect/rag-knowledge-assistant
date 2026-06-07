import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from chromadb.config import Settings

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

UPLOAD_DIR = "data"
DB_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #020617 100%);
}

.hero-card {
    padding: 34px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(37,99,235,0.20), rgba(147,51,234,0.14));
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 28px;
}

.hero-title {
    font-size: 46px;
    font-weight: 900;
    line-height: 1.1;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 18px;
    margin-top: 14px;
    max-width: 920px;
}

.metric-row {
    display: flex;
    gap: 16px;
    margin-top: 24px;
    flex-wrap: wrap;
}

.metric-card {
    padding: 14px 18px;
    border-radius: 16px;
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e5e7eb;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin: 26px 0 14px 0;
}

.chat-card {
    padding: 22px;
    border-radius: 22px;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(148,163,184,0.18);
    margin-bottom: 20px;
}

.user-label {
    color: #93c5fd;
    font-weight: 800;
    margin-bottom: 6px;
}

.ai-label {
    color: #86efac;
    font-weight: 800;
    margin-top: 18px;
    margin-bottom: 6px;
}

.source-card {
    padding: 14px;
    border-radius: 14px;
    background: rgba(2,6,23,0.85);
    border: 1px solid rgba(148,163,184,0.14);
    margin-bottom: 12px;
}

.stButton > button {
    border-radius: 14px;
    padding: 0.65rem 1rem;
    font-weight: 700;
}

.stTextInput > div > div > input {
    border-radius: 14px;
}

.small-muted {
    color: #94a3b8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


def reset_knowledge_base():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.session_state.chat_history = []
    st.session_state.processed_files = set()


def load_documents(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def create_vector_db(file_path):
    docs = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.warning("No readable text found in this document.")
        return None

    settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True
    )

    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR,
            client_settings=settings
        )
        return vector_db

    except Exception as e:
        st.warning("Vector database reset required. Rebuilding knowledge base...")
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR,
            client_settings=settings
        )
        return vector_db


def get_vector_db():
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )


def generate_answer(question, context):
    prompt = f"""
You are a professional Multi-PDF RAG Knowledge Assistant.

You can answer questions using context retrieved from multiple uploaded documents.

Rules:
- Answer only using the uploaded document context.
- If multiple documents contain relevant information, combine the answer clearly.
- Keep the answer clear, concise, and professional.
- If the answer is not present in the context, say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def export_chat():
    output = "Multi-PDF RAG Knowledge Assistant - Chat History\n\n"
    for chat in st.session_state.chat_history:
        output += f"Question: {chat['question']}\n"
        output += f"Answer: {chat['answer']}\n"
        output += "-" * 60 + "\n"
    return output


with st.sidebar:
    st.markdown("## 📚 Multi-Document Knowledge Base")
    st.caption("Upload multiple PDFs or TXT files and search across all of them.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    create_vector_db(file_path)

                st.session_state.processed_files.add(uploaded_file.name)

        st.success("Documents indexed successfully.")

    st.divider()

    st.markdown("### 📄 Indexed Documents")
    if st.session_state.processed_files:
        for file in sorted(st.session_state.processed_files):
            st.markdown(f"✅ `{file}`")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear KB"):
            reset_knowledge_base()
            st.success("Knowledge base cleared.")

    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat cleared.")

    if st.session_state.chat_history:
        st.download_button(
            "Download Chat",
            data=export_chat(),
            file_name="multi_pdf_rag_chat_history.txt",
            mime="text/plain"
        )


st.markdown("""
<div class="hero-card">
    <div class="hero-title">📚 Multi-PDF RAG Knowledge Assistant</div>
    <div class="hero-subtitle">
        Upload multiple PDFs or TXT files and ask questions across all documents using semantic search, vector embeddings, and LLM-powered answer generation.
    </div>
    <div class="metric-row">
        <div class="metric-card">📄 Multi-Document Upload</div>
        <div class="metric-card">🧠 Hugging Face Embeddings</div>
        <div class="metric-card">🗂️ ChromaDB Vector Search</div>
        <div class="metric-card">⚡ Groq LLM</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section-title">Ask Across Your Documents</div>', unsafe_allow_html=True)

suggested_questions = [
    "Summarize all uploaded documents.",
    "What are the key points from these documents?",
    "Compare the uploaded documents.",
    "List important facts mentioned in the documents."
]

cols = st.columns(4)
selected_question = ""

for index, item in enumerate(suggested_questions):
    with cols[index]:
        if st.button(item, use_container_width=True):
            selected_question = item

question = st.text_input(
    "Enter your question",
    value=selected_question,
    placeholder="Example: Summarize the main points from all uploaded documents."
)

ask = st.button("Generate Answer", use_container_width=False)

if ask:
    if not question:
        st.warning("Please enter a question.")
    elif not os.path.exists(DB_DIR):
        st.warning("Please upload and index at least one document first.")
    elif not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Add it in Streamlit Secrets.")
    else:
        try:
            vector_db = get_vector_db()
            docs = vector_db.similarity_search(question, k=4)

            if not docs:
                st.warning("No relevant context found in uploaded documents.")
            else:
                context = "\n\n".join([doc.page_content for doc in docs])

                with st.spinner("Retrieving relevant context from uploaded documents..."):
                    answer = generate_answer(question, context)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": docs
                })

        except Exception as e:
            st.error("Something went wrong while searching the knowledge base.")
            st.info("Click 'Clear KB', upload the documents again, and retry.")


st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.info("Upload one or more documents and ask a question to start.")

for chat in reversed(st.session_state.chat_history):
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)

    st.markdown('<div class="user-label">User Question</div>', unsafe_allow_html=True)
    st.write(chat["question"])

    st.markdown('<div class="ai-label">AI Answer</div>', unsafe_allow_html=True)
    st.write(chat["answer"])

    with st.expander("View retrieved document sources"):
        for i, doc in enumerate(chat["sources"], start=1):
            page = doc.metadata.get("page", "Unknown")
            source = os.path.basename(doc.metadata.get("source", "Unknown"))

            st.markdown(
                f"""
                <div class="source-card">
                    <b>Source {i}</b><br>
                    <span class="small-muted">Document:</span> {source}<br>
                    <span class="small-muted">Page:</span> {page}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(doc.page_content)

    st.markdown("</div>", unsafe_allow_html=True)