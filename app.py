import ollama
import pypdf
import streamlit as st
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

warnings.filterwarnings("ignore")

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
        div[data-testid="stStatusWidget"] {visibility: hidden; height: 0%;}
        .stDeployButton {display:none;}
        .stChatMessage {padding: 0.5rem; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

TARGET_MODEL = "phi3:latest"
TOP_K_CHUNKS = 5
TOP_K_CHUNKS_LIST_QUERY = 18     # wider retrieval for "list all / every / each" style questions
CHUNK_SIZE = 1000
MAX_PDF_PAGES = 200
HISTORY_TURNS = 2                # kept small to speed up generation
MAX_ANSWER_TOKENS = 512          # caps generation time so answers finish faster
FULL_CONTEXT_WORD_LIMIT = 3000   # below this, skip retrieval and use the whole document as context

LIST_QUERY_KEYWORDS = ["all ", "list", "every", "each ", "8 ", "eight", "complete list", "summarize"]

# ==========================================
# 2. LAZY-LOAD RESOURCES
# ==========================================
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def check_ollama_model(model_name: str) -> bool:
    try:
        ollama.show(model_name)
        return True
    except Exception:
        return False

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF. Returns (pages_text, error_message)."""
    try:
        reader = pypdf.PdfReader(uploaded_file)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return [], "This PDF is password-protected and could not be opened."

        if len(reader.pages) > MAX_PDF_PAGES:
            return [], f"This PDF has {len(reader.pages)} pages, which exceeds the {MAX_PDF_PAGES}-page limit for this demo."

        pages_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages_text.append((i + 1, page_text))

        return pages_text, None
    except Exception as e:
        return [], f"Could not read this PDF: {e}"

def split_text_with_pages(pages_text, chunk_size: int = CHUNK_SIZE, overlap: int = 150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = []
    chunk_index = 0
    for page_num, page_text in pages_text:
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            # chunk_index preserves the chunk's original position in the document,
            # so retrieved results can later be re-sorted into reading order.
            documents.append(Document(
                page_content=chunk,
                metadata={"page": page_num, "chunk_index": chunk_index}
            ))
            chunk_index += 1
    return documents

def is_list_style_query(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in LIST_QUERY_KEYWORDS)

def build_context_text(context_docs: list) -> tuple:
    """Re-sort retrieved chunks into their original document order (not similarity-score
    order). This keeps multi-part sections (like a numbered list or sequential narrative
    spanning several pages) coherent, instead of the model seeing fragments jumbled and
    stitching them together incorrectly. Returns (context_text, pages_used)."""
    ordered_docs = sorted(context_docs, key=lambda d: d.metadata.get("chunk_index", 0))
    context_text = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in ordered_docs
    )
    pages_used = sorted({doc.metadata.get("page", "?") for doc in ordered_docs})
    return context_text, pages_used

def generate_rag_response(question: str, context_text: str, chat_history: list):
    system_prompt = (
        "You are a helpful assistant answering questions about a PDF document.\n"
        "STRICT RULES:\n"
        "1. Only use facts explicitly present in the context below. Do not add, infer, "
        "or combine information that isn't directly stated.\n"
        "2. The context may contain multiple distinct sections or labeled parts (e.g. a skills "
        "table, a numbered question list, a learning plan, or sequential stages/years). Treat each "
        "section or label strictly on its own — never attach a detail to the wrong label just "
        "because it appears nearby, never treat a row from a table as an item from a different "
        "numbered list, and never merge content from unrelated sections into one point.\n"
        "3. If asked to list or summarize items in order (e.g. 'all 8 questions', 'each year'), "
        "only include what is explicitly presented under that exact item in the context. Do not "
        "invent an item to fill a gap, do not repeat an earlier item to pad the count, and do not "
        "borrow a sentence from the next or previous item.\n"
        "4. If the full answer is not present in the context, say so honestly instead of guessing "
        "or completing the pattern yourself.\n"
        "5. If the context is insufficient, reply exactly: 'The document does not provide this information.'\n\n"
        f"Context:\n{context_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-(HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": question})

    try:
        response_stream = ollama.chat(
            model=TARGET_MODEL,
            messages=messages,
            stream=True,
            options={"num_predict": MAX_ANSWER_TOKENS},
        )
        for chunk in response_stream:
            yield chunk["message"]["content"]
    except Exception:
        yield (
            f"⚠️ Backend connection failed. Please ensure `ollama serve` is running "
            f"and that the `{TARGET_MODEL}` model is pulled (`ollama pull {TARGET_MODEL}`)."
        )

# ==========================================
# 4. SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "full_pages_text" not in st.session_state:
    st.session_state.full_pages_text = None
if "ollama_ready" not in st.session_state:
    st.session_state.ollama_ready = None

# ==========================================
# 5. SIDEBAR UI
# ==========================================
with st.sidebar:
    st.title("📄 PDF Assistant")
    st.markdown("Chat securely offline with your documents.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your document", type=["pdf"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if uploaded_file:
                st.session_state.messages = [
                    {"role": "assistant", "content": f"Chat cleared! What else would you like to know about **{uploaded_file.name}**?"}
                ]
            else:
                st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            transcript = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
            )
            st.download_button(
                "⬇️", data=transcript, file_name="chat_transcript.txt",
                use_container_width=True, help="Download chat transcript"
            )

# ==========================================
# 6. STARTUP CHECK
# ==========================================
if st.session_state.ollama_ready is None:
    st.session_state.ollama_ready = check_ollama_model(TARGET_MODEL)

if not st.session_state.ollama_ready:
    st.error(
        f"⚠️ Model **{TARGET_MODEL}** was not found, or the Ollama server isn't running.\n\n"
        f"Run these in your terminal, then reload this page:\n"
        f"```\nollama serve\nollama pull {TARGET_MODEL}\n```"
    )
    st.stop()

# ==========================================
# 7. MAIN CHAT UI
# ==========================================
if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.current_file_id != file_id:
        st.session_state.current_file_id = file_id

        # ---- Single, clean, non-technical loading indicator ----
        with st.spinner("📖 Reading your document..."):
            pages_text, error = extract_text_from_pdf(uploaded_file)

            if not error and pages_text:
                documents = split_text_with_pages(pages_text)
                embeddings = get_embeddings_model()
                st.session_state.vectorstore = FAISS.from_documents(documents, embeddings)
                st.session_state.full_pages_text = pages_text

        if error:
            st.error(error)
            st.session_state.vectorstore = None
            st.session_state.current_file_id = None
        elif not pages_text:
            st.error("No readable text found in this PDF. It may be a scanned/image-only document.")
            st.session_state.vectorstore = None
            st.session_state.current_file_id = None
        else:
            st.toast("Document ready!", icon="✅")
            st.session_state.messages = [
                {"role": "assistant", "content": f"✅ I have successfully read **{uploaded_file.name}**. What would you like to know about it?"}
            ]

# Empty State UI
if not uploaded_file:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Chat with any PDF</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray; font-weight: normal;'>Upload a document in the sidebar to get started.</h4>", unsafe_allow_html=True)

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about the document..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            if st.session_state.vectorstore is None:
                st.write("Please wait for the document to finish processing.")
            else:
                # ---- Highly visible "thinking" indicator so the app never looks frozen ----
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("🧠 *Searching document...*")

                list_style = is_list_style_query(prompt)
                full_pages_text = st.session_state.full_pages_text or []
                total_words = sum(len(t.split()) for _, t in full_pages_text)

                # For short documents on list/summary-style questions, skip retrieval entirely
                # and feed the whole document as one continuous block. This avoids similarity
                # search selecting only a subset of a sequential narrative (e.g. missing a
                # section) and avoids chunk-boundary fragmentation causing the model to blend
                # details from adjacent sections/labels together.
                if list_style and total_words <= FULL_CONTEXT_WORD_LIMIT and full_pages_text:
                    context_text = "\n\n".join(
                        f"[Page {page_num}]\n{page_text}" for page_num, page_text in full_pages_text
                    )
                    pages_used = sorted({p for p, _ in full_pages_text})
                else:
                    k = TOP_K_CHUNKS_LIST_QUERY if list_style else TOP_K_CHUNKS
                    retrieved_docs = st.session_state.vectorstore.similarity_search(prompt, k=k)
                    context_text, pages_used = build_context_text(retrieved_docs)

                history_for_model = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]

                thinking_placeholder.markdown("💬 *Generating answer (this can take a moment)...*")
                response_generator = generate_rag_response(prompt, context_text, history_for_model)

                # Clear the placeholder right as streaming starts so text flows in cleanly
                first_chunk_container = st.empty()
                full_response = ""
                for i, token in enumerate(response_generator):
                    if i == 0:
                        thinking_placeholder.empty()
                    full_response += token
                    first_chunk_container.markdown(full_response + "▌")
                first_chunk_container.markdown(full_response)

                if pages_used:
                    with st.expander("📌 Sources used for this answer"):
                        st.write(", ".join(f"Page {p}" for p in pages_used))

                st.session_state.messages.append({"role": "assistant", "content": full_response})
