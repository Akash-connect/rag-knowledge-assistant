# 📚 Multi-PDF RAG Knowledge Assistant

A premium **Retrieval-Augmented Generation (RAG)** application that enables users to upload multiple PDF and TXT documents and ask intelligent questions across all uploaded files using **LangChain, FAISS, Hugging Face Embeddings, and Groq LLM**.

---

## 🚀 Live Demo

🌐 **Live Application**

[https://your-streamlit-app.streamlit.app](https://rag-knowledge-assistant-leqlw4ar2xvvc62j3yemnc.streamlit.app/)

---

## 📷 Preview

### Features

- 📄 Upload multiple PDF and TXT documents
- 🧠 Semantic Search with FAISS
- 🤖 Groq LLM-powered Answer Generation
- 🔍 Hugging Face Embeddings
- 📚 Multi-Document Retrieval
- 💬 Chat History
- 📌 Source Document & Page Citation
- ⬇️ Download Conversation
- 🎨 Modern Premium Streamlit UI
- ☁️ Free Deployment on Streamlit Cloud

---

## 🏗️ System Architecture

```text
                Upload Documents
                       │
                       ▼
              Document Loader
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
         Hugging Face Embeddings
                       │
                       ▼
               FAISS Vector Store
                       │
                       ▼
            Semantic Similarity Search
                       │
                       ▼
               Retrieved Context
                       │
                       ▼
                   Groq LLM
                       │
                       ▼
                Generated Answer
                       │
                       ▼
                 Streamlit UI
```

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | Hugging Face |
| LLM | Groq (Llama 3.1) |
| Document Loader | PyPDF |
| Deployment | Streamlit Cloud |

---

## 📂 Project Structure

```text
rag-knowledge-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .python-version
├── .gitignore
├── data/
└── faiss_index/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

For Streamlit Cloud:

```toml
GROQ_API_KEY="your_groq_api_key"
```

---

## 📋 Sample Questions

### General

- Summarize all uploaded documents.
- What are the key points from these documents?
- Compare the uploaded documents.
- List important facts mentioned in the documents.

### Employee Policy

- What is the leave policy?
- How many paid leaves are provided?
- What is the probation period?

### Product Manual

- What is the warranty period?
- How do I reset the device?

### University Guide

- What attendance percentage is required?
- How many credits are needed?

### Travel Policy

- What is the reimbursement limit?
- What expenses are covered?

---

## 📦 Requirements

```text
streamlit==1.41.1
langchain==0.3.14
langchain-community==0.3.14
langchain-text-splitters==0.3.5
faiss-cpu==1.9.0.post1
sentence-transformers==3.3.1
pypdf==5.1.0
python-dotenv==1.0.1
groq==0.13.1
```

---

## ✨ Key Features

### Multi-Document Retrieval

Upload multiple PDFs and ask questions across all documents.

### Semantic Search

Uses Hugging Face embeddings and FAISS vector search for highly relevant document retrieval.

### AI-Powered Question Answering

Groq LLM generates context-aware responses from retrieved information.

### Source Citation

Displays the source document and page number used to generate the answer.

### Premium UI

Modern and responsive Streamlit interface with:

- Sidebar document manager
- Chat history
- Suggested questions
- Download conversation
- Clear knowledge base

---

## 🚀 Deployment

### Streamlit Cloud

1. Push project to GitHub.
2. Create a new Streamlit app.
3. Connect GitHub repository.
4. Add GROQ_API_KEY in Secrets.
5. Deploy.

---

## 💼 Resume Description

### Multi-PDF RAG Knowledge Assistant | Python, LangChain, FAISS, Hugging Face, Groq, Streamlit

- Built and deployed an end-to-end Multi-PDF Retrieval-Augmented Generation (RAG) application capable of answering domain-specific questions from uploaded documents.
- Developed intelligent document ingestion pipelines with text chunking, vector embeddings, and semantic search.
- Integrated FAISS for efficient vector storage and similarity retrieval.
- Leveraged Hugging Face embedding models and Groq LLM for context-aware answer generation.
- Designed a modern Streamlit interface supporting multi-document upload, conversational querying, and source attribution.
- Deployed the application on Streamlit Cloud for public access and demonstration.

---

## 🎯 Future Improvements

- Conversation Memory
- OCR Support
- Image Understanding
- Multi-Modal RAG
- Voice Input
- Authentication
- Admin Dashboard
- Cloud Storage Integration
- API Support
- Docker Deployment

---

## 👨‍💻 Author

# Akash Jadhav

AI/ML Engineer 


## ⭐ If you found this project useful, please consider giving it a star.
