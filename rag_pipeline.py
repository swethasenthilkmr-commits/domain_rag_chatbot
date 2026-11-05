"""
rag_pipeline.py
Module 5 + 6: retrieve the most relevant chunks and generate a grounded answer.

question -> embedding -> FAISS search (top-k chunks)
         -> context + question -> Gemini (strict prompt) -> answer + sources
"""

import os
import time

from langchain_core.output_parsers import StrOutputParser
from prompt import rag_prompt, FALLBACK_MESSAGE

# Gemini model (can be changed in .env, e.g. LLM_MODEL=gemini-3.8-flash).
# Free-tier quotas are per model, so switching models helps when one is used up.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
TOP_K = 4                 # guideline: retrieve the top 3-5 chunks
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.2"))   # below this a chunk is not relevant


def get_llm(api_key):
    """Connect Google Gemini through LangChain."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    # max_retries / timeout: if Gemini is busy, give up after about a minute
    # instead of retrying for several minutes (the app then shows a "busy" message)
    return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key,
                                  max_retries=1, timeout=45)


def format_context(results):
    """Join the retrieved chunks, each labelled with its document and page."""
    parts = []
    for doc, score in results:
        label = f"[Source: {doc.metadata['source']}, page {doc.metadata['page']}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def unique_sources(results):
    """List of unique (document, page) pairs with the best score and a short snippet."""
    seen = {}
    for doc, score in results:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key not in seen or score > seen[key]["score"]:
            seen[key] = {"source": key[0], "page": key[1], "score": round(score, 3),
                         "snippet": doc.page_content[:300].replace("\n", " ") + "..."}
    return sorted(seen.values(), key=lambda s: -s["score"])


def answer_question(question, store, embeddings, llm, k=TOP_K):
    """Run the full RAG pipeline for one question.

    Returns a dict: answer, sources, found (True if answered from the documents), seconds.
    """
    start = time.time()

    # Module 5: retrieval
    results = store.search(question, embeddings, k=k)
    relevant = [(doc, score) for doc, score in results if score >= MIN_SIMILARITY]

    # nothing relevant enough -> do not ask the LLM, return the fallback message
    if not relevant:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "found": False,
                "retrieved": unique_sources(results), "seconds": round(time.time() - start, 2)}

    # Module 6: generation with the strict prompt
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(relevant), "question": question}).strip()

    found = FALLBACK_MESSAGE.lower() not in answer.lower()
    return {
        "answer": answer,
        "sources": unique_sources(relevant) if found else [],
        "found": found,
        "retrieved": unique_sources(results),
        "seconds": round(time.time() - start, 2),
    }
