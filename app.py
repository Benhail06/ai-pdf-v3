import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Load key from Streamlit Cloud if deployed
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="AI PDF Assistant V3", layout="wide")

st.title("📄 AI PDF Assistant (V3)")
st.caption("Upload a PDF and ask questions from it using Groq LLM.")

with st.sidebar:
    st.header("Controls")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []




# Session memory
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_db = FAISS.from_documents(docs, embeddings)

    st.success("Document processed successfully!")

if st.session_state.vector_db is None:
    st.info("Please upload a PDF to begin.")
    st.stop()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)



retriever = st.session_state.vector_db.as_retriever()

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []


user_input = st.chat_input("Ask a question about the document")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {user_input}
        """

        response = llm.invoke(prompt).content

    st.session_state.chat_history.append(("assistant", response))


for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
