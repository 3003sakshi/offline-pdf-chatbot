# 📄 Offline PDF Chatbot

A fully offline, privacy-preserving RAG (Retrieval-Augmented Generation) chatbot that lets you
upload a PDF and ask questions about it — no internet connection, cloud API, or external service
required at query time.

Built with **Streamlit**, **FAISS** (vector search), **HuggingFace sentence-transformers**
(local embeddings), and **Ollama** running **phi3** (local LLM inference).

---

## ✨ Features

- Upload any text-based PDF and chat with it in natural language
- Runs 100% locally/offline — no data ever leaves your machine
- Source attribution: shows which page(s) an answer was drawn from
- Handles encrypted/corrupted PDFs and oversized documents gracefully
- Query-aware retrieval: automatically widens context for "list all / summarize each" style
  questions, and falls back to feeding the *entire* document for short files, to avoid missing
  or blending information across sections
- Short conversational memory so follow-up questions work
- Clear "model not ready" screen if Ollama/phi3 isn't running, instead of a silent crash

---

## 🧠 Tech Stack (LLM & RAG Framework)

For quick reference in a report or viva:

| Component | What we used |
|---|---|
| **LLM (generation)** | `phi3:latest` (Microsoft Phi-3, ~3.8B parameters), run **locally via Ollama** |
| **RAG framework / orchestration** | LangChain (`langchain-text-splitters` for chunking) |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs on CPU) |
| **Vector store / retriever** | FAISS (Facebook AI Similarity Search), in-memory, per session |
| **PDF parsing** | `pypdf` |
| **UI** | Streamlit |

This is a full **offline Retrieval-Augmented Generation (RAG)** pipeline — the same architectural
pattern used by cloud RAG products, just with every component (LLM, embeddings, vector index)
running locally instead of calling an external API. No document text or query ever leaves the
machine.

---

## 🧱 Architecture

```
PDF upload
   │
   ▼
Text extraction (pypdf) ──► per-page text
   │
   ▼
Chunking (LangChain RecursiveCharacterTextSplitter, 1000 chars, 150 overlap)
   │
   ▼
Embeddings (sentence-transformers/all-MiniLM-L6-v2, CPU) ──► 384-dim vectors
   │
   ▼
Vector index (FAISS, in-memory, per-session)
   │
   ▼
User question ──► similarity search (or full-document context for short PDFs on
                   list/summary questions) ──► phi3 via Ollama (streamed) ──► answer
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally

### 2. Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Pull the local model
```bash
ollama pull phi3:latest
```

### 4. Start Ollama's server (in a separate terminal)
```bash
ollama serve
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will refuse to proceed (with clear instructions) if Ollama or the `phi3` model isn't
available, instead of failing silently.

---

## 🧪 How it was evaluated

The app was tested against multiple real PDFs with a mix of question types to check for
faithfulness (i.e. that answers are grounded in the document and not hallucinated):

| Test type | Example question | Purpose |
|---|---|---|
| Basic fact retrieval | "When and where was X born?" | Sanity check |
| Specific numeric detail | "How much rent did X pay?" | Tests chunk precision |
| Enumeration | "List all 8 interview questions" / "list all his habits" | Tests whether the model invents/drops/merges items — a common small-LLM failure mode |
| Chronological summary | "Summarize what happened in each of the 4 years" | Tests whether details get misattributed to the wrong section |
| Negative test | "What college did he attend for his Master's?" (not in the document) | Confirms the model says *"The document does not provide this information"* instead of hallucinating |

### Known limitation
Because this project intentionally uses a small (~3.8B parameter), CPU-only, fully local model
(`phi3`) rather than a large cloud model, it can occasionally misattribute a detail to the wrong
labeled section (e.g. attaching a sentence from "Year 2" to "Year 1" in a summarization task),
even though it correctly refuses to answer when information is genuinely absent. This is a
known trade-off of prioritizing full offline privacy over model size, and is mitigated (not
fully eliminated) by:
- reordering retrieved chunks back into document order before prompting,
- widening retrieval for list/summary-style questions,
- falling back to full-document context for short PDFs, and
- explicit prompt instructions not to blend adjacent sections.

---

## 📁 Project structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Possible future improvements
- Support scanned/image-only PDFs via OCR
- Persist vector index to disk so re-uploading the same file skips re-embedding
- Multi-document chat (query across several PDFs at once)
- Configurable model selection (swap `phi3` for another locally pulled Ollama model)
