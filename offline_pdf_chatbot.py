import streamlit as st
import pypdf
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("📄 PDF Chatbot (Offline)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# -------- SIMPLE TEXT SPLITTER --------
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text = text.strip()

    if len(text) == 0:
        return chunks

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
# ------------------------------------

@st.cache_data
def extract_text(file):
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -------- HELPER: BEST SENTENCE --------
def extract_best_answer(docs, question):
    question_words = set(re.findall(r"\w+", question.lower()))
    scored_sentences = []

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        for sent in sentences:
            sent_words = set(re.findall(r"\w+", sent.lower()))
            score = len(question_words & sent_words)
            if score > 0:
                scored_sentences.append((score, sent.strip()))

    if not scored_sentences:
        return "Answer not clearly found in the PDF."

    scored_sentences.sort(reverse=True)
    return scored_sentences[0][1]
# -------------------------------------

# -------- SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
# -------------------------------

if uploaded_file:
    pdf_text = extract_text(uploaded_file)

    if len(pdf_text.strip()) == 0:
        st.error("❌ No readable text found in this PDF.")
        st.stop()

    if not st.session_state.pdf_ready:
        chunks = split_text(pdf_text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.pdf_ready = True

    st.success("PDF processed successfully!")

# -------- CHAT UI --------
if st.session_state.pdf_ready:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about the PDF")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching PDF..."):
                docs = st.session_state.vectorstore.similarity_search(
                    question, k=5
                )

                answer = extract_best_answer(docs, question)
                st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload a text-based PDF to start chatting.")
