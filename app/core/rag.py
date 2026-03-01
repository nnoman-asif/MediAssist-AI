import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.utils.logger import get_logger

log = get_logger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        model_dir = str(config.MODELS_DIR / "all-MiniLM-L6-v2")
        log.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=model_dir,
            model_kwargs={"device": "cpu"},
        )
        log.info("Embedding model loaded")
    return _embeddings


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def load_vectorstore() -> Optional[FAISS]:
    """Load an existing FAISS index from disk, if present."""
    global _vectorstore
    index_path = config.VECTORSTORE_DIR / "index.faiss"
    if index_path.exists():
        log.info("Loading existing vector store from %s", config.VECTORSTORE_DIR)
        _vectorstore = FAISS.load_local(
            str(config.VECTORSTORE_DIR),
            embeddings=_get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        log.info("Vector store loaded (%d vectors)", _vectorstore.index.ntotal)
        return _vectorstore
    log.info("No existing vector store found")
    return None


def index_pdf(pdf_path: str | Path) -> int:
    """Index a single PDF into the vector store. Returns number of chunks added."""
    global _vectorstore
    pdf_path = Path(pdf_path)
    log.info("Indexing PDF: %s", pdf_path.name)

    start = time.perf_counter()
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    chunks = _get_splitter().split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source_filename"] = pdf_path.name

    embeddings = _get_embeddings()

    if _vectorstore is None:
        _vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        new_store = FAISS.from_documents(chunks, embeddings)
        _vectorstore.merge_from(new_store)

    _vectorstore.save_local(str(config.VECTORSTORE_DIR))
    elapsed = time.perf_counter() - start
    log.info("Indexed %d chunks from %s in %.1f s", len(chunks), pdf_path.name, elapsed)
    return len(chunks)


def rebuild_from_pdfs(pdf_dir: str | Path) -> int:
    """Drop the existing index and re-index all PDFs in a directory."""
    global _vectorstore
    _vectorstore = None
    pdf_dir = Path(pdf_dir)
    total = 0
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        total += index_pdf(pdf)
    return total


def search(query: str, k: int = 8) -> list[dict]:
    """Similarity search. Returns list of {content, metadata, score}."""
    if _vectorstore is None:
        load_vectorstore()
    if _vectorstore is None:
        return []

    results = _vectorstore.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        output.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(float(score), 4),
            }
        )
    return output


def remove_by_filename(filename: str) -> bool:
    """Remove all chunks from a specific PDF and rebuild. Returns True if store was modified."""
    global _vectorstore
    if _vectorstore is None:
        load_vectorstore()
    if _vectorstore is None:
        return False

    remaining_pdfs = [
        p for p in config.PDFS_DIR.glob("*.pdf") if p.name != filename
    ]

    _vectorstore = None
    if remaining_pdfs:
        for pdf in remaining_pdfs:
            index_pdf(pdf)
        return True

    if config.VECTORSTORE_DIR.exists():
        for f in config.VECTORSTORE_DIR.iterdir():
            f.unlink()
    return True


def get_store_size() -> int:
    if _vectorstore is None:
        load_vectorstore()
    if _vectorstore is None:
        return 0
    return _vectorstore.index.ntotal
