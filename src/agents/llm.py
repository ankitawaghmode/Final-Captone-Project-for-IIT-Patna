import os

from langchain_core.language_models import BaseChatModel


def create_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Return a chat LLM for the given provider (openai | gemini)."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def create_embeddings(provider: str | None = None, model: str | None = None):
    """Return embeddings for the given provider (openai | gemini)."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
