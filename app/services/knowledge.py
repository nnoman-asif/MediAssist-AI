import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from app import config
from app.core import rag
from app.utils.logger import get_logger

log = get_logger(__name__)

MANIFEST_PATH = config.PDFS_DIR / "manifest.json"


def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_pdf(source_path: str | Path) -> dict:
    """Copy a PDF into data/pdfs/, index it, and track in manifest.

    Returns dict with keys: filename, status, chunks.
    """
    source = Path(source_path)
    if not source.exists():
        return {"filename": source.name, "status": "error", "chunks": 0}

    file_hash = _file_hash(source)
    manifest = _load_manifest()

    for entry in manifest:
        if entry.get("hash") == file_hash:
            log.info("PDF '%s' already indexed (hash match)", source.name)
            return {"filename": source.name, "status": "already_indexed", "chunks": entry.get("chunks", 0)}

    dest = config.PDFS_DIR / source.name
    if dest.exists():
        stem = source.stem
        suffix = source.suffix
        dest = config.PDFS_DIR / f"{stem}_{file_hash[:8]}{suffix}"

    shutil.copy2(str(source), str(dest))
    log.info("Copied PDF to %s", dest)

    num_chunks = rag.index_pdf(dest)

    manifest.append(
        {
            "filename": dest.name,
            "hash": file_hash,
            "chunks": num_chunks,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_manifest(manifest)

    return {"filename": dest.name, "status": "indexed", "chunks": num_chunks}


def list_documents() -> list[dict]:
    """Return all indexed PDFs with metadata."""
    return _load_manifest()


def delete_document(filename: str) -> bool:
    """Remove a PDF and its chunks from the knowledge base."""
    manifest = _load_manifest()
    found = False

    new_manifest = []
    for entry in manifest:
        if entry["filename"] == filename:
            found = True
            pdf_path = config.PDFS_DIR / filename
            if pdf_path.exists():
                pdf_path.unlink()
                log.info("Deleted PDF file %s", filename)
        else:
            new_manifest.append(entry)

    if not found:
        log.warning("PDF '%s' not found in manifest", filename)
        return False

    _save_manifest(new_manifest)
    rag.remove_by_filename(filename)
    log.info("Removed '%s' from knowledge base", filename)
    return True


def rebuild_index() -> int:
    """Drop existing index and re-index all PDFs."""
    log.info("Rebuilding full vector store index ...")
    total = rag.rebuild_from_pdfs(config.PDFS_DIR)
    log.info("Rebuild complete: %d total chunks", total)
    return total

