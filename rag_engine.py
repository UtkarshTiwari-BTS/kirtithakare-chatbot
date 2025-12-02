from azure_client import client

AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"

def retrieve(query, index, vectorizer, chunks, k=3):
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    distances, ids = index.search(query_vec, k)

    context = "\n\n".join(chunks[i] for i in ids[0])
    return context


def ask_llm(question, context):
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
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
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content



#Previous 
"""from azure_client import client

AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"
def retrieve(query, index, vectorizer, chunks, k=3):
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    distances, ids = index.search(query_vec, k)

    context = "\n\n".join(chunks[i] for i in ids[0])
    return context

def ask_llm(question, context):
    response = client.chat.completions.create(
        model= AZURE_OPENAI_DEPLOYMENT,
          temperature=0.2, 
        messages=[
            {"role": "system", "content": "Answer ONLY using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content
"""
