import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os

st.set_page_config(
    page_title="AI Document Q&A Agent",
    page_icon="🤖",
    layout="centered"
)

# ── Gemini setup ─────────────────────────────────────────────────────────────
def configure_gemini(api_key):
    genai.configure(api_key=api_key)

# ── Extract text from PDFs ────────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# ── Split text into chunks ────────────────────────────────────────────────────
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# ── Create FAISS vector store ─────────────────────────────────────────────────
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ── QA chain with Gemini ──────────────────────────────────────────────────────
def get_qa_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say:
    "The answer is not available in the uploaded document."
    Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ── Answer question ───────────────────────────────────────────────────────────
def answer_question(question, api_key):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=4)
    chain = get_qa_chain(api_key)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 AI-Powered Document Q&A Agent")
st.markdown("Upload PDFs and ask questions — powered by **Google Gemini + RAG**.")
st.markdown("---")

# Sidebar for API key and PDF upload
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("🔑 Enter Gemini API Key", type="password",
                             help="Get your free key at aistudio.google.com")
    st.markdown("---")
    st.header("📄 Upload Documents")
    pdf_docs = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process Documents", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key first.")
        elif not pdf_docs:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Processing documents..."):
                configure_gemini(api_key)
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("Could not extract text from the PDFs.")
                else:
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success(f"✅ {len(pdf_docs)} document(s) processed! ({len(chunks)} chunks)")

# Main — question answering
st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here...",
                          placeholder="e.g. What is the main topic of this document?")

if st.button("Get Answer", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif not os.path.exists("faiss_index"):
        st.warning("Please upload and process documents first.")
    else:
        with st.spinner("Thinking..."):
            try:
                configure_gemini(api_key)
                answer = answer_question(question, api_key)
                st.markdown("### 📝 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown(
    "<small>Built by **Abhradeep Chandra Paul** · "
    "[GitHub](https://github.com/abhra-deep) · "
    "[LinkedIn](https://linkedin.com/in/abhradeepchandrapaul)</small>",
    unsafe_allow_html=True
)
