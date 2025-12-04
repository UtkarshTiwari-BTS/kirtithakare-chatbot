# rag_engine.py
import azure_client
def retrieve_and_ask(query, chunks, index, vectorizer, k=3):
    # Step 1: Retrieve top-k relevant chunks
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    distances, ids = index.search(query_vec, k)
    context = "\n\n".join(chunks[i] for i in ids[0])

    # Step 2: Ask Azure OpenAI
    try:
        response = azure_client.client.chat.completions.create(
            model=azure_client.deployment_model,  # <-- Use model instead of deployment_id
            temperature=0.2,
             messages=[
            {
                "role": "system",
                "content":
                (
                    "You are a strict RAG assistant.\n"
                    "Answer ONLY using the provided context.\n"
                    "RULES:\n"
                    "1. Do NOT use outside knowledge.\n"
                    "2. Do NOT guess or hallucinate.\n"
                    "3. If information is missing, say: 'The answer is not available in the provided context.'\n"
                    "4. Keep answers short, clear, and factual.\n"
                )
            },
             {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }

        ]

        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error communicating with Azure OpenAI: {str(e)}"
