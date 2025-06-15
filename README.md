# RAG-Powered Document Summarizer 🧠📄

A Flask-based web app that summarizes uploaded documents (PDF, TXT, MD) using a Retrieval-Augmented Generation (RAG) pipeline built on:
- **FAISS** for similarity search
- **HuggingFace Transformers** for summarization (`facebook/bart-large-cnn`)
- **LangChain** for integration and orchestration

---

## 🚀 Features

- Upload and process multiple documents at once
- Extracts and chunks text from PDFs, Markdown, and Text files
- Embeds documents using `all-MiniLM-L6-v2` (via `sentence-transformers`)
- Retrieves top-k relevant chunks with FAISS similarity search
- Summarizes them using BART (`facebook/bart-large-cnn`)
- CLI logs: token count, latency, retrieved chunks

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/abdullahjaved36603/text_summarizer.git
   cd text_summarizer


