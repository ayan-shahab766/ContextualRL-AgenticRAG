"""ChromaDB-backed vector store.

Chroma / LangChain are imported lazily so that merely importing this module (or
anything that transitively imports it) does not require the heavy native stack.
Our lightweight :class:`~contextual_rl_agentic_rag.schema.Document` is converted to/from
LangChain documents only at the boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from contextual_rl_agentic_rag.retrieval.base import BaseVectorStore, content_hash
from contextual_rl_agentic_rag.schema import Document


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, chroma_dir: str | Path, embeddings) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embeddings = embeddings
        self._lc_embeddings = None
        self._stores: dict = {}

    def _langchain_embeddings(self):
        """Wrap the backend in a LangChain ``Embeddings`` subclass for compatibility."""
        if self._lc_embeddings is None:
            backend = self._embeddings
            try:
                from langchain_core.embeddings import Embeddings

                class _Adapter(Embeddings):
                    def embed_documents(self, texts):
                        return backend.embed_documents(list(texts))

                    def embed_query(self, text):
                        return backend.embed_query(text)

                self._lc_embeddings = _Adapter()
            except Exception:  # noqa: BLE001 - fall back to duck typing
                self._lc_embeddings = backend
        return self._lc_embeddings

    @staticmethod
    def _to_lc(doc: Document):
        from langchain_core.documents import Document as LCDocument

        return LCDocument(page_content=doc.page_content, metadata=doc.metadata)

    @staticmethod
    def _from_lc(lc) -> Document:
        return Document(page_content=lc.page_content, metadata=dict(lc.metadata or {}))

    def get_store(self, collection: str):
        if collection not in self._stores:
            from langchain_chroma import Chroma

            self._stores[collection] = Chroma(
                client=self.client,
                collection_name=collection,
                embedding_function=self._langchain_embeddings(),
            )
        return self._stores[collection]

    def add_documents(self, collection: str, docs: list[Document]) -> int:
        if not docs:
            return 0
        store = self.get_store(collection)
        store.add_documents(
            [self._to_lc(d) for d in docs],
            ids=[content_hash(d.page_content) for d in docs],
        )
        return len(docs)

    def search(
        self, collection: str, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ) -> list[Document]:
        store = self.get_store(collection)
        kwargs: dict = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        return [self._from_lc(d) for d in store.similarity_search(query, **kwargs)]

    def search_with_scores(
        self, collection: str, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        store = self.get_store(collection)
        return [
            (self._from_lc(d), float(s))
            for d, s in store.similarity_search_with_relevance_scores(query, k=k)
        ]

    def list_collections(self) -> list[dict]:
        return [{"name": c.name, "count": c.count()} for c in self.client.list_collections()]
