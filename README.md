# 📄 Offline PDF Chatbot

An **offline AI-powered PDF chatbot** that allows users to ask questions directly from PDF documents **without using the internet or external APIs**. The system is designed for **privacy-sensitive and restricted environments**, where data sovereignty is critical.

---

## 🚀 Features

* Fully **offline** operation (no cloud APIs)
* Upload and process **text-based PDFs**
* Semantic search using **vector embeddings**
* Fast similarity search with **FAISS**
* Simple and interactive **Streamlit chat UI**

---

## 🧠 How It Works (Architecture Overview)

1. User uploads a PDF file
2. Text is extracted from the PDF using `pypdf`
3. Extracted text is split into overlapping chunks
4. Each chunk is converted into vector embeddings using a local embedding model
5. FAISS stores and searches vectors based on semantic similarity
6. The most relevant text chunks are selected
7. A heuristic-based method extracts the best matching sentence as the answer

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – User interface
* **PyPDF** – PDF text extraction
* **Sentence-Transformers** – Text embeddings
* **FAISS** – Vector similarity search
* **LangChain Community** – Embedding and vector store wrappers

---

## 📌 Project Status

✅ Core functionality implemented and working
⚠️ Answer accuracy may vary for complex or multi-context questions

This project focuses on **offline architecture and secure deployment** rather than perfect answer accuracy.

---

## ⚠️ Known Limitations

* Accuracy depends on chunk size and overlap strategy
* No re-ranking or fine-tuning applied
* Works best with **text-based PDFs** (not scanned images)
* Long or highly technical documents may reduce precision

---

## 🔮 Future Improvements

* Improved text chunking strategy
* Context re-ranking for better retrieval accuracy
* Integration of a local generative language model
* Evaluation metrics for answer quality

---

## 🔐 Why Offline?

* Ensures complete **data privacy**
* Suitable for **defence, research, and restricted environments**
* No dependency on third-party APIs or internet connectivity

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run offline_pdf_chatbot.py
```

---

## 📂 Folder Structure

```
offline-pdf-chatbot/
│
├── offline_pdf_chatbot.py
├── requirements.txt
├── README.md
└── data/ (optional sample PDFs)
```

---

## 👩‍💻 Author

**Sakshi Shekhawat**
B.E. Student | AI & Data Science
Interested in AI, NLP, and privacy-focused systems

---

## 📄 License

This project is intended for educational and learning purposes.
