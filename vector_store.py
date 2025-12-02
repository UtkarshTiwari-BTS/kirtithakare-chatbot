import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vector_store(chunks):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks).toarray().astype("float32")

    dim = matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    return index, vectorizer, matrix
