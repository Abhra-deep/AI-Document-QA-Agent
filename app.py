import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import google.generativeai as genai
import os

st.set_page_config(page_title="AI Document Q&A Agent", page_icon="🤖", layout="centered")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def answer_question(question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
Answer the question from the provided context only.
If the answer is not in the context, say "The answer is not available in the uploaded document."
Do not make up answers.

Context: {context}
Question: {input}
Answer:""")
    retriever = db.as_retriever(search_kwargs={"k": 4})
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

# UI
st.title("🤖 AI-Powered Document Q&A Agent")
st.markdown("Upload PDFs and ask questions — powered by **Google Gemini + RAG**.")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("🔑 Enter Gemini API Key", type="password",
                             help="Get your free key at aistudio.google.com")
    st.markdown("---")
    st.header("📄 Upload Documents")
    pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process Documents", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key first.")
        elif not pdf_docs:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Processing documents..."):
                genai.configure(api_key=api_key)
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("Could not extract text from the PDFs.")
                else:
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks, api_key)
                    st.success(f"✅ {len(pdf_docs)} document(s) processed! ({len(chunks)} chunks)")

st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here...", placeholder="e.g. What is the main topic of this document?")

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
                answer = answer_question(question, api_key)
                st.markdown("### 📝 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("<small>Built by **Abhradeep Chandra Paul** · [GitHub](https://github.com/abhra-deep) · [LinkedIn](https://linkedin.com/in/abhradeepchandrapaul)</small>", unsafe_allow_html=True)
