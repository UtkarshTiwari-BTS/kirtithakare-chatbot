
import streamlit as st
from pdf_reader import read_pdf
from chunker import chunk_text
from vector_store import build_vector_store
from rag_engine import retrieve, ask_llm

# ---------------------------
# Initialize history
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📘 PDF RAG Chatbot with History")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "BajajTravelsInsurance.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    index, vectorizer, _ = build_vector_store(chunks)

    st.success("PDF processed successfully!")

    # -------------- CHAT HISTORY --------------
    st.subheader("Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**🧑‍💻 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")

    # -------------- USER QUESTION --------------
    question = st.text_input("Ask a question about the PDF")

    if question:
        context = retrieve(question, index, vectorizer, chunks)
        answer = ask_llm(question, context)

        st.session_state.history.append((question, answer))

        st.subheader("Answer:")
        st.write(answer)
