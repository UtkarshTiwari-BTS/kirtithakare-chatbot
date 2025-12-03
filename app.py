"""import streamlit as st
from pdf_reader import read_pdf
from chunker import chunk_text
from vector_store import build_vector_store
from rag_engine import retrieve, ask_llm

st.title(" PDF RAG Chatbot ")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "BajajTravelsInsurance.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    #st.info("Reading PDF...")
    text = read_pdf(pdf_path)

    #st.info("Chunking text...")
    chunks = chunk_text(text)

    #st.info("Building TF-IDF + FAISS index...")
    index, vectorizer, _ = build_vector_store(chunks)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question about the PDF")

    if question:
        #st.info("Retrieving relevant context...")
        context = retrieve(question, index, vectorizer, chunks)

        #st.info("Querying Azure OpenAI...")
        answer = ask_llm(question, context)

        st.subheader("Answer:")
        st.write(answer)
"""

#Store history
import streamlit as st
from pdf_reader import read_pdf
from chunker import chunk_text
from vector_store import build_vector_store
from rag_engine import retrieve, ask_llm


# ---------------------------------------------
# Initialize chat history
# ---------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


st.title("PDF RAG Chatbot with History")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "BajajTravelsInsurance.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    index, vectorizer, _ = build_vector_store(chunks)

    st.success("PDF processed successfully!")

    # ---------------------------------------------
    # Display Chat History
    # ---------------------------------------------
    st.subheader("Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**🧑‍💻 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")

    # ---------------------------------------------
    # Ask Question Input
    # ---------------------------------------------
    question = st.text_input("Ask a question about the PDF")

    if question:
        context = retrieve(question, index, vectorizer, chunks)
        answer = ask_llm(question, context)

        # Save Q&A to history
        st.session_state.history.append((question, answer))

        # Display Latest Answer
        st.subheader("Answer:")
        st.write(answer)

