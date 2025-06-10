# 📚 GenAI Document Chatbot

An AI-powered chatbot that lets you upload PDFs and ask questions about their content — all **offline**, using local language models like TinyLlama via Ollama, powered by LangChain and FAISS.


## Features
- ✅ Ask questions across multiple PDF documents
- ✅ Uses local LLMs (no OpenAI API key needed!)
- ✅ Automatically extracts, chunks, and embeds text
- ✅ Chat memory to maintain conversation context
- ✅ Cleans up vector store after each session
- ✅ Resilient to crashes, slow queries, and bad input

## Tech Stack
- Python
- LangChain
- FAISS (for semantic search)
- HuggingFace Embeddings
- Ollama + TinyLlama (local LLM)
- PyMuPDF (fitz) (PDF extraction)









