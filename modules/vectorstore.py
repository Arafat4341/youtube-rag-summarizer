from langchain_community.vectorstores import FAISS

def build_faiss_index(text_chunks, embeddings):
    """Creates a FAISS vector index from text chunks."""
    return FAISS.from_texts(text_chunks, embeddings)

def retrieve_similar(query, faiss_index, top_k=3):
    """Finds similar content using FAISS index."""
    return faiss_index.similarity_search(query, k=top_k)
