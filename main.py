#This file only  debugging purpose 
import sys
from pdf_reader import read_pdf
from chunker import chunk_text
from vector_store import build_vector_store
from rag_engine import retrieve, ask_llm

def log(msg):
    print("\n========== DEBUG ==========")
    print(msg)
    print("===========================\n")
    sys.stdout.flush()


def main():
    print("\n=== PDF RAG Chatbot Backend ===\n")

    pdf_path = "uploaded.pdf"   # You can change this

    # -------------------------
    # READ PDF
    # -------------------------
    print("Reading PDF...")
    text = read_pdf(pdf_path)
    log(f"PDF LENGTH: {len(text)} characters")

    # -------------------------
    # CHUNKING
    # -------------------------
    print("Chunking text...")
    chunks = chunk_text(text)
    log(f"CHUNKS CREATED: {len(chunks)}")

    # -------------------------
    # TF-IDF + FAISS INDEX
    # -------------------------
    print("Building TF-IDF + FAISS index...")
    index, vectorizer, embeddings = build_vector_store(chunks)
    log("Index created successfully!")

    print("\nBackend is ready! Ask questions.\n")
    print("Type 'exit' to quit.\n")

    # -------------------------
    # CHAT LOOP (Terminal)
    # -------------------------
    while True:
        question = input("\nYOU: ")
    
        if question.lower() in ["exit", "quit"]:
            print("Exiting chatbot...")
            break

        log(f"QUESTION: {question}")

        # Retrieve relevant context
        context = retrieve(question, index, vectorizer, chunks)
        log(f"RETRIEVED CONTEXT:\n{context[:500]} ...")

        # Get LLM answer
        answer = ask_llm(question, context)
        log(f"ANSWER GENERATED")

        print(f"\nBOT: {answer}")



if __name__ == "__main__":
    main()
