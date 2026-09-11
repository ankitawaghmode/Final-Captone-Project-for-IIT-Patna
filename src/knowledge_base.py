"""
RAG knowledge base using ChromaDB with pluggable embeddings (OpenAI or Gemini).
"""
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

KB_JSON    = Path(__file__).parent.parent / "data" / "kb_documents.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


def _load_documents() -> list[Document]:
    with open(KB_JSON, encoding="utf-8") as f:
        articles = json.load(f)

    return [
        Document(
            page_content=(
                f"Title: {a['title']}\n"
                f"Category: {a['category']}\n\n"
                f"{a['content']}"
            ),
            metadata={
                "id":       a["id"],
                "title":    a["title"],
                "category": a["category"],
                "keywords": ", ".join(a.get("keywords", [])),
            },
        )
        for a in articles
    ]


def init_knowledge_base(embeddings: Embeddings, collection_name: str = "it_knowledge_base") -> Chroma:
    """Load or build the persistent ChromaDB vector store.
    Use a distinct collection_name per embedding provider to avoid dimension mismatches.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if vectorstore._collection.count() == 0:
        docs = _load_documents()
        vectorstore.add_documents(docs)
        print(f"✓ Indexed {len(docs)} KB articles into ChromaDB")
    else:
        print(f"✓ KB already loaded ({vectorstore._collection.count()} articles)")

    return vectorstore


def search_kb(vectorstore: Chroma, query: str, k: int = 2) -> list[dict]:
    """Return top-k KB articles relevant to the query."""
    hits = vectorstore.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in hits:
        content = doc.page_content
        # Strip the title/category header so only the article body is returned
        if "\n\n" in content:
            content = content.split("\n\n", 1)[1]
        results.append(
            {
                "title":    doc.metadata.get("title", ""),
                "category": doc.metadata.get("category", ""),
                "content":  content,
                "score":    round(float(score), 4),
            }
        )
    return results
