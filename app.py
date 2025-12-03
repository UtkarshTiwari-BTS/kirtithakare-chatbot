import streamlit as st
import azure_client
from pdf_reader import read_pdf
from chunker import chunk_text
from vector_store import build_vector_store
from rag_engine import retrieve_and_ask

st.set_page_config(page_title="PDF RAG Chatbot ", layout="wide")
st.title(" PDF RAG Chatbot (Azure OpenAI)")

# ---------------- Sidebar ----------------
st.sidebar.header(" Azure OpenAI Credentials")
api_key = st.sidebar.text_input("API Key", type="password")
endpoint = st.sidebar.text_input("Endpoint (e.g. https://YOUR_RESOURCE.openai.azure.com/)")
deployment = st.sidebar.text_input("Deployment Name (e.g. gpt-4o-mini)")

if not api_key or not endpoint or not deployment:
    st.sidebar.warning("Please enter Azure API Key, Endpoint, and Deployment Name.")
    st.stop()

azure_client.init_azure_client(api_key, endpoint, deployment)
# ---------------- Chat History ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PDF Upload ----------------
st.subheader(" Upload PDF")
uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    pdf_path = "BajajTravelsInsurance.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("PDF uploaded successfully!")

    # Read PDF
    text = read_pdf(pdf_path)

    # Chunk text
    chunks = chunk_text(text)

    # Build vector store
    index, vectorizer = build_vector_store(chunks)

    st.success(f"PDF processed. Total chunks: {len(chunks)}")

    # ---------------- Chat History Display ----------------
    st.subheader(" Chat History")
    if st.session_state.history:
        for q, a in st.session_state.history:
            st.markdown(f" You: {q}")
            st.markdown(f" Bot: {a}")
            st.write("---")
    else:
        st.info("No messages yet.")

    # ---------------- Ask Question ----------------
    st.subheader("Ask a question about the PDF")
    question = st.text_input("Your Question:")

    if question:
        answer = retrieve_and_ask(
            query=question,
            chunks=chunks,
            index=index,
            vectorizer=vectorizer
        )
        st.session_state.history.append((question, answer))

        st.subheader(" Answer")
        st.write(answer)
