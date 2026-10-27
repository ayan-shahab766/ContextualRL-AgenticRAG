"""File processors: parse, chunk, and prepare files for indexing.

Heavy parsers (LangChain loaders, pandas) are imported lazily inside ``process``
so importing this module stays cheap. All processors emit lightweight
:class:`~contextual_rl_agentic_rag.schema.Document` objects.
"""
from __future__ import annotations

import base64
import re
from typing import Optional

from contextual_rl_agentic_rag.config import Settings, settings as default_settings
from contextual_rl_agentic_rag.llm.base import LLMBackend
from contextual_rl_agentic_rag.schema import Document


def _to_doc(lc, extra: Optional[dict] = None) -> Document:
    md = dict(getattr(lc, "metadata", {}) or {})
    if extra:
        md.update(extra)
    return Document(page_content=lc.page_content, metadata=md)


class PDFProcessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings

    def process(self, file_path: str, metadata: dict) -> dict:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        pages = PyPDFLoader(file_path).load()
        for p in pages:
            p.metadata.update(metadata)

        standard_chunks = [_to_doc(c) for c in splitter.split_documents(pages)]

        sentence_chunks: list[Document] = []
        window = self.settings.sentence_window_size
        for page in pages:
            sentences = re.split(r"(?<=[.!?])\s+", page.page_content)
            for i, sent in enumerate(sentences):
                if not sent.strip():
                    continue
                w_start = max(0, i - window)
                w_end = min(len(sentences), i + window + 1)
                sentence_chunks.append(
                    Document(
                        page_content=sent.strip(),
                        metadata={
                            **metadata,
                            "window_text": " ".join(sentences[w_start:w_end]),
                            "sentence_index": i,
                            "page": page.metadata.get("page", 0),
                        },
                    )
                )

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
        raw_parents = parent_splitter.split_documents(pages)
        parent_chunks: list[Document] = []
        child_chunks: list[Document] = []
        for i, parent in enumerate(raw_parents):
            parent_id = f"parent_{metadata.get('file_id', '')}_{i}"
            parent.metadata["parent_id"] = parent_id
            parent_chunks.append(_to_doc(parent))
            for child in child_splitter.split_documents([parent]):
                child_chunks.append(
                    _to_doc(child, {"parent_id": parent_id, "parent_content": parent.page_content})
                )

        full_text = "\n".join(p.page_content for p in pages)
        return {
            "standard_chunks": standard_chunks,
            "sentence_chunks": sentence_chunks,
            "parent_chunks": parent_chunks,
            "child_chunks": child_chunks,
            "full_text": full_text,
            "file_type": "pdf",
        }


class ImageProcessor:
    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm

    def process(self, file_path: str, metadata: dict) -> dict:
        with open(file_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        description = self.llm.vision(
            img_b64,
            "Describe this image in comprehensive detail for search indexing. "
            "Include: objects, text/labels, colors, layout, data shown, relationships between elements.",
        )

        doc = Document(
            page_content=description,
            metadata={
                **metadata,
                "image_path": file_path,
                "modality": "image",
                "image_b64_preview": img_b64[:500],
            },
        )
        return {
            "standard_chunks": [doc],
            "sentence_chunks": [doc],
            "child_chunks": [doc],
            "parent_chunks": [doc],
            "full_text": description,
            "file_type": "image",
        }


class CSVProcessor:
    def process(self, file_path: str, metadata: dict) -> dict:
        import pandas as pd

        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

        schema = (
            f"TABLE: {metadata.get('source_filename', 'data')}\n"
            f"COLUMNS: {', '.join(map(str, df.columns))}\n"
            f"ROWS: {len(df)} | DTYPES: {dict(df.dtypes.astype(str))}\n"
            f"SAMPLE:\n{df.head(5).to_string()}\n"
            f"STATS:\n{df.describe(include='all').to_string()}"
        )
        schema_doc = Document(
            page_content=schema,
            metadata={
                **metadata,
                "chunk_type": "schema",
                "columns": list(map(str, df.columns)),
                "row_count": len(df),
                "csv_path": file_path,
            },
        )

        row_chunks: list[Document] = []
        for i in range(0, len(df), 25):
            group = df.iloc[i : i + 25]
            row_chunks.append(
                Document(
                    page_content=(
                        f"Rows {i}-{i + len(group)} of {metadata.get('source_filename', 'data')}:\n"
                        f"{group.to_string()}"
                    ),
                    metadata={**metadata, "chunk_type": "rows", "row_start": i},
                )
            )

        all_chunks = [schema_doc] + row_chunks
        return {
            "standard_chunks": all_chunks,
            "sentence_chunks": all_chunks,
            "child_chunks": all_chunks,
            "parent_chunks": [schema_doc],
            "full_text": schema,
            "file_type": "csv",
            "csv_path": file_path,
        }
