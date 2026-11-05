"""
vector_store.py
Module 4 and 5: create embeddings, store them in FAISS, save/load the index,
and search for the chunks most similar to a question.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 numbers per text).
Embeddings are normalised, so the FAISS inner product = cosine similarity.
"""

import json
import os

import faiss
import numpy as np
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def get_embedding_model():
    """Load the pretrained Sentence Transformers model through LangChain.

    The model (about 90 MB) is downloaded automatically the first time.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},   # needed for cosine similarity
    )


class VectorStore:
    """A small FAISS vector store: the FAISS index + the chunk texts and metadata."""

    def __init__(self, index, chunks):
        self.index = index      # FAISS index with one vector per chunk
        self.chunks = chunks    # list of LangChain Documents in the same order

    @classmethod
    def build(cls, chunks, embeddings):
        """Convert every chunk into an embedding and store it in FAISS."""
        vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]), dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])   # IP = inner product = cosine for unit vectors
        index.add(vectors)
        return cls(index, chunks)

    def search(self, question, embeddings, k=4):
        """Return the top-k chunks for a question as a list of (Document, similarity score)."""
        query = np.array([embeddings.embed_query(question)], dtype="float32")
        faiss.normalize_L2(query)
        scores, ids = self.index.search(query, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, folder):
        """Save the FAISS index and the chunk texts/metadata to a folder."""
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, INDEX_FILE))
        data = [{"text": c.page_content, "metadata": c.metadata} for c in self.chunks]
        with open(os.path.join(folder, CHUNKS_FILE), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, folder):
        """Load a saved index (plain JSON + FAISS file, no pickle)."""
        index = faiss.read_index(os.path.join(folder, INDEX_FILE))
        with open(os.path.join(folder, CHUNKS_FILE), encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Document(page_content=d["text"], metadata=d["metadata"]) for d in data]
        return cls(index, chunks)

    @staticmethod
    def exists(folder):
        return (os.path.exists(os.path.join(folder, INDEX_FILE))
                and os.path.exists(os.path.join(folder, CHUNKS_FILE)))

    def document_names(self):
        return sorted({c.metadata["source"] for c in self.chunks})
