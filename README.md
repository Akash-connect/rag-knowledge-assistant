# 📚 Multi-PDF RAG Knowledge Assistant

A premium **Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF/TXT documents and ask intelligent questions across all uploaded files.

## 🚀 Live Demo

🔗 https://rag-knowledge-assistant-leqlw4ar2xvvc62j3yemnc.streamlit.app/

## 📌 Features

- Multi-PDF and TXT document upload
- Document chunking using LangChain
- Hugging Face sentence embeddings
- FAISS vector search
- Groq LLM-powered answer generation
- Source document and page reference
- Chat history
- Download conversation
- Clean Streamlit UI
- Free deployment on Streamlit Cloud

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- Groq LLM
- PyPDF

## 🧠 Architecture

```text
PDF/TXT Upload
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Hugging Face Embeddings
      ↓
FAISS Vector Store
      ↓
Semantic Similarity Search
      ↓
Retrieved Context
      ↓
Groq LLM
      ↓
Generated Answer
      ↓
Streamlit UI
📂 Project Structure
rag-knowledge-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .python-version
├── .gitignore
└── data/
⚙️ Installation
git clone https://github.com/your-username/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
streamlit run app.py
🔐 Environment Variable

Create a .env file locally:

GROQ_API_KEY=your_groq_api_key_here

For Streamlit Cloud, add this in Secrets:

GROQ_API_KEY = "your_groq_api_key_here"
📋 Sample Questions
Summarize all uploaded documents.
What are the key points from these documents?
Compare the uploaded documents.
List important facts mentioned in the documents.
Which document contains warranty information?
What policy is mentioned about reimbursement?
📦 requirements.txt
streamlit==1.41.1
langchain==0.3.14
langchain-community==0.3.14
langchain-text-splitters==0.3.5
faiss-cpu==1.9.0.post1
sentence-transformers==3.3.1
pypdf==5.1.0
python-dotenv==1.0.1
groq==0.13.1
👨‍💻 Author

Akash Jadhav

⭐ Resume Highlight

Built and deployed a Multi-PDF RAG Knowledge Assistant using LangChain, FAISS, Hugging Face embeddings, Groq LLM, and Streamlit, enabling semantic search and context-aware question answering across multiple uploaded documents.
