import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="AI Document Q&A Agent", page_icon="🤖", layout="centered")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def answer_question(question, context, api_key):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided document context.
If the answer is not in the document, say "The answer is not available in the uploaded document."
Do not make up answers.

DOCUMENT CONTEXT:
{context[:50000]}

QUESTION: {question}

ANSWER:"""
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content

# UI
st.title("🤖 AI-Powered Document Q&A Agent")
st.markdown("Upload PDFs and ask questions — powered by **Google Gemini**.")
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
            with st.spinner("Reading documents..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("Could not extract text from the PDFs.")
                else:
                    st.session_state['doc_text'] = raw_text
                    st.success(f"✅ {len(pdf_docs)} document(s) processed! Ready for questions.")

st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here...",
                          placeholder="e.g. What is the main topic of this document?")

if st.button("Get Answer", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif 'doc_text' not in st.session_state:
        st.warning("Please upload and process documents first.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(question, st.session_state['doc_text'], api_key)
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
