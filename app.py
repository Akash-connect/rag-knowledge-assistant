import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

UPLOAD_DIR = "data"
DB_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 10px;
}
.subtitle {
    font-size: 18px;
    color: #c9c9c9;
    margin-bottom: 30px;
}
.chat-box {
    padding: 18px;
    border-radius: 14px;
    background-color: #1f2937;
    margin-bottom: 16px;
}
.user-msg {
    color: #93c5fd;
    font-weight: 700;
}
.ai-msg {
    color: #86efac;
    font-weight: 700;
}
.source-box {
    background-color: #111827;
    padding: 14px;
    border-radius: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


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

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return vector_db


def get_vector_db():
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


def generate_answer(question, context):
    prompt = f"""
You are a helpful RAG-based AI knowledge assistant.

Answer the question only using the uploaded document context.
If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

Keep the answer clear, simple, and professional.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def export_chat():
    text = "RAG Knowledge Assistant - Chat History\n\n"

    for chat in st.session_state.chat_history:
        text += f"Question: {chat['question']}\n"
        text += f"Answer: {chat['answer']}\n"
        text += "-" * 60 + "\n"

    return text


with st.sidebar:
    st.header("📚 Knowledge Base")

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

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    create_vector_db(file_path)

                st.session_state.processed_files.add(uploaded_file.name)

        st.success("All documents processed successfully!")

    st.divider()

    st.subheader("📄 Processed Files")

    if st.session_state.processed_files:
        for file in st.session_state.processed_files:
            st.write(f"✅ {file}")
    else:
        st.write("No files uploaded yet.")

    st.divider()

    if st.button("🗑️ Clear Knowledge Base"):
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.success("Knowledge base cleared successfully!")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared successfully!")

    if st.session_state.chat_history:
        st.download_button(
            label="⬇️ Download Chat",
            data=export_chat(),
            file_name="rag_chat_history.txt",
            mime="text/plain"
        )


st.markdown(
    """
    <div class="main-title">🤖 RAG-Based LLM Knowledge Assistant</div>
    <div class="subtitle">
    Upload documents and ask intelligent, context-aware questions using LangChain, ChromaDB, Hugging Face embeddings, and Groq LLM.
    </div>
    """,
    unsafe_allow_html=True
)

suggested_questions = [
    "How many paid leaves are allowed annually?",
    "How many sick leaves are provided?",
    "What is the leave policy?",
    "Summarize this document.",
]

st.subheader("💡 Suggested Questions")

cols = st.columns(4)

selected_question = None

for i, question_text in enumerate(suggested_questions):
    with cols[i]:
        if st.button(question_text):
            selected_question = question_text

question = st.text_input(
    "Ask a question from your documents",
    value=selected_question if selected_question else ""
)

if st.button("🚀 Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    elif not os.path.exists(DB_DIR):
        st.warning("Please upload and process documents first.")
    elif not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    else:
        vector_db = get_vector_db()

        docs = vector_db.similarity_search(question, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        with st.spinner("Searching documents and generating answer..."):
            answer = generate_answer(question, context)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": docs
        })


st.divider()

st.subheader("💬 Chat History")

if not st.session_state.chat_history:
    st.info("Ask your first question to start the conversation.")

for chat in reversed(st.session_state.chat_history):
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    st.markdown('<div class="user-msg">👤 You</div>', unsafe_allow_html=True)
    st.write(chat["question"])

    st.markdown('<div class="ai-msg">🤖 AI Assistant</div>', unsafe_allow_html=True)
    st.write(chat["answer"])

    with st.expander("📌 Retrieved Sources"):
        for i, doc in enumerate(chat["sources"], start=1):
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Unknown")

            st.markdown(
                f"""
                <div class="source-box">
                <b>Source {i}</b><br>
                File: {source}<br>
                Page: {page}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(doc.page_content)

    st.markdown("</div>", unsafe_allow_html=True)