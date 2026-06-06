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
    layout="wide"
)

UPLOAD_DIR = "data"
DB_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def load_documents(file_path):
    if file_path.endswith(".pdf"):
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
You are a helpful AI knowledge assistant.

Answer the user's question only using the uploaded document context.
If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

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

with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Processing {uploaded_file.name}..."):
                create_vector_db(file_path)

        st.success("All documents processed successfully!")

    st.divider()

    if st.button("Clear Knowledge Base"):
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            st.session_state.chat_history = []
            st.success("Knowledge base cleared successfully!")
        else:
            st.warning("No knowledge base found.")

st.markdown("""
# RAG-Based LLM Knowledge Assistant

Upload multiple documents and ask context-aware questions using Retrieval-Augmented Generation.
""")

question = st.text_input("Ask a question from your documents")

if st.button("Get Answer"):
    if question:
        if not os.path.exists(DB_DIR):
            st.warning("Please upload and process documents first.")
        else:
            vector_db = get_vector_db()
            docs = vector_db.similarity_search(question, k=3)

            context = "\n\n".join([doc.page_content for doc in docs])

            with st.spinner("Generating answer..."):
                answer = generate_answer(question, context)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": docs
            })
    else:
        st.warning("Please enter a question.")

for chat in reversed(st.session_state.chat_history):
    st.markdown("---")

    st.markdown("### Question")
    st.write(chat["question"])

    st.markdown("### Answer")
    st.write(chat["answer"])

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(chat["sources"], start=1):
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Unknown")

            st.markdown(f"**Source {i} | Page: {page}**")
            st.caption(source)
            st.write(doc.page_content)