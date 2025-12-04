import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vector_store(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray().astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, vectorizer
