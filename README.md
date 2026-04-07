# RAG CLI: Saarland University Module Handbook Assistant

A fully **local, offline** Retrieval-Augmented Generation (RAG) system for querying Saarland University's Computer related MSc course module handbooks.

**Ask questions like:**
- "Which modules cover cryptography?"
- "Tell me about Machine Learning courses"
- "What are the prerequisites for Neural Networks?"
- "How many ECTS is the Master Thesis?"

Get instant, contextual answers powered by local AI—no API keys, no internet required, no rate limits.

---

## 🎯 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- ~2GB disk space for models

### 1. Clone or Download
```bash
git clone <repo-url>
cd rag-cli
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama (just 2 minutes)
Ollama runs local LLMs on your computer.

**macOS/Linux:**
```bash
# Via official installer
https://ollama.com/download

# Or Homebrew
brew install ollama
```

**Windows:**
Download from https://ollama.com/download

### 4. Pull a Model
```bash
# Start Ollama in background
ollama serve

# In another terminal, pull a model:
ollama pull llama3.2      # Recommended (2GB, great quality)
# or
ollama pull llama3.2:1b   # Fast (1GB, lightweight)
```

### 5. Index Your Module Handbook 
This step is **only required if you are adding new PDFs**.  
The existing `chroma_db` already contains indexed data for previously ingested PDFs.

```bash
python rag.py ingest data/dsai.pdf
```

### 6. Start Chatting!
```bash
# Interactive mode
python rag.py chat

# Or ask a single question
python rag.py ask "Which modules cover cryptography?"
```

That's it! Everything runs locally on your machine.

---

## 🧠 What is RAG?

**RAG** = **R**etrieval-**A**ugmented **G**eneration

Instead of asking an LLM to generate answers from its training data (which may be outdated or hallucinate), RAG:

1. **Retrieves** relevant documents from your knowledge base
2. **Augments** the LLM prompt with this context
3. **Generates** an answer based on the retrieved context

**Advantage:** Answers are grounded in your actual documents, not AI imagination.

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG CLI System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Input (Questions)                                     │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CLI Interface (rag.py)                              │   │
│  │  - ingest (index PDFs)                               │   │
│  │  - ask (single question)                             │   │
│  │  - chat (interactive)                                │   │
│  │  - status (show info)                                │   │
│  └──────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Processing Pipeline                                 │   │
│  │                                                      │   │
│  │  INPUT → EXTRACT → CHUNK → EMBED → STORE             │   │
│  │                                                      │   │
│  │  ├─ PDF Extraction (pdfplumber)                      │   │
│  │  ├─ Text Chunking (1200 chars, 300 overlap)          │   │
│  │  ├─ Embedding (sentence-transformers)                │   │
│  │  └─ Storage (ChromaDB)                               │   │
│  └──────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐   │ 
│  │  Vector Database (ChromaDB)                          │   │
│  │  - 456 chunks from handbook                          │   │
│  │  - Cosine similarity search                          │   │
│  │  - Persistent storage in ./chroma_db                 │   │
│  └──────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Query → Embed → Search → Retrieve                   │   │
│  │  Top-8 relevant chunks returned                      │   │
│  └──────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LLM Inference (Ollama)                              │   │
│  │  - Sends context + question to local LLM             │   │
│  │  - Streams response in real-time                     │   │
│  │  - No internet, no API calls                         │   │
│  └──────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  Answer to User                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Ollama + llama3.2 | Local, free, 2GB, great quality |
| **Embeddings** | sentence-transformers | Fast, accurate, 438MB |
| **Vector DB** | ChromaDB | Lightweight, persistent, no server |
| **PDF Parsing** | pdfplumber | Accurate text extraction |
| **Language** | Python 3.8+ | Easy to modify, great libraries |

---

**Available Models:**

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| llama3.2 | 2GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | `ollama pull llama3.2` |
| llama3.2:1b | 1GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | `ollama pull llama3.2:1b` |
| mistral | 4GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | `ollama pull mistral` |
| phi3 | 2GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `ollama pull phi3` |
| gemma2:2b | 1.6GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `ollama pull gemma2:2b` |

---

## 🚀 Usage

### Command Overview

```bash
python rag.py <command> [options]

# Available commands:
python rag.py setup                    # Show setup guide
python rag.py ingest <pdf_files>      # Index PDF(s)
python rag.py ask "<question>"        # Ask a single question
python rag.py chat                    # Interactive mode
python rag.py status                  # Show system status
```

### Ask a Single Question

```bash
python rag.py ask "Which modules cover Machine Learning?"

# Output:
# INFO     | Retrieving top-8 chunks...
# INFO     | Found 8 relevant chunks (avg score: 0.823)
# Model: llama3.2
#
# ======================================================================
# ANSWER
# ======================================================================
# According to the handbook, the following modules cover Machine Learning:
#
# 1. Machine Learning (ML) - 9 ECTS
#    - Part of core DSAI curriculum
#    - Covers Bayesian decision theory, kernel methods, model selection
#
# 2. Neural Networks: Theory and Implementation (NNTI) - 9 ECTS
#    - Includes deep learning and neural network implementations
#
# [More content...]
#
# ======================================================================
# SOURCES USED
# ======================================================================
#   • dsai.pdf  (similarity: 0.89)
```

---

### Interactive Chat Mode

```bash
python rag.py chat

# Output:
# ======================================================================
#   Saarland University Module Handbook — RAG Assistant
#   Type 'quit' to exit. 'help' for options.
# ======================================================================
#
# You: Tell me about cryptography modules
# INFO     | Retrieving top-8 chunks...
# INFO     | Found 8 relevant chunks (avg score: 0.856)
# Model: llama3.2
#
# ======================================================================
# ANSWER
# ======================================================================
# The Saarland University offers several modules related to cryptography:
#
# 1. **Cryptography (Crypto)** - 9 ECTS
#    Semester: 1
#    ...
#
# ======================================================================
# SOURCES USED
# ======================================================================
#   • dsai.pdf  (similarity: 0.91)
#
# You: What are the prerequisites?
# [Next answer...]
#
# You: quit
# Goodbye!
```

**Interactive commands:**
- `quit` or `exit` - Exit chat
- Any question - Ask about modules

---

### Check System Status

```bash
python rag.py status

# Output:
#
#   ChromaDB
#     Path       : /Users/you/project/chroma_db
#     Collection : module_handbooks
#     Documents  : 456
#     Files      : 1
#       • dsai.pdf                            ( 456 chunks)
#
#   Ollama
#     Running    : ✓ Yes
#     Models     : llama3.2, mistral
#     Active     : llama3.2
#
```

---

## 🐛 Known Issues & Limitations

### Current Limitations

1. **English only:** Optimized for English text (works with German but less accurate)
2. **No metadata filtering:** Can't filter by module type or semester
3. **No conversation memory:** Each query is independent (no multi-turn context)
4. **Limited to local**: Works offline but can't access web resources

### Known Issues

| Issue | Impact | Workaround |
|-------|--------|-----------|
| Very large PDFs (>500 pages) | Slow indexing | Split into multiple files |
| Offline Ollama | Can't generate answers | Must have Ollama running |
| Corrupted PDFs | Indexing fails | Use pdftotext to validate |
| Very specific queries | May return generic chunks | Rephrase question more broadly |

---

