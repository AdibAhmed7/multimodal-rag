"""Document Manager for Streamlit UI.

Handles PDF file upload, local storage, end-to-end processing execution
(ingestion -> chunking -> embedding -> indexing), and active document discovery.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DocumentManager:
    """Manages document uploads, pipeline processing, and document registry."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT
        self.raw_dir = self.root / "data" / "raw" / "pdfs"
        self.metadata_dir = self.root / "data" / "processed" / "metadata"
        self.chunks_dir = self.root / "data" / "processed" / "chunks"
        self.embeddings_dir = self.root / "data" / "processed" / "embeddings"
        self.qdrant_dir = self.root / "data" / "processed" / "qdrant"

        # Ensure base directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def run_script(self, script_name: str, *args) -> None:
        """Executes a pipeline script via subprocess from the project root."""
        script_path = self.root / "scripts" / script_name
        command = [sys.executable, str(script_path)] + [str(a) for a in args]
        result = subprocess.run(
            command,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Execution of {script_name} failed:\n{err}")

    def save_uploaded_file(self, uploaded_file) -> Path:
        """Saves an uploaded Streamlit file buffer into data/raw/pdfs/."""
        dest_path = self.raw_dir / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return dest_path

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Runs the four-stage indexing pipeline for a given PDF document."""
        document_id = pdf_path.stem

        with st.status(f"Processing '{pdf_path.name}'...", expanded=True) as status:
            # Step 1: Ingestion and extraction
            st.write("Step 1/4: Ingesting PDF & extracting figures...")
            self.run_script("ingest_pdf.py", str(pdf_path))
            metadata_path = self.metadata_dir / f"{document_id}.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            # Step 2: Semantic chunking and linking
            st.write("Step 2/4: Chunking text & linking figures...")
            self.run_script("prepare_chunks.py", str(metadata_path))
            doc_chunks_dir = self.chunks_dir / document_id
            if not doc_chunks_dir.exists():
                raise FileNotFoundError(f"Chunk directory not found: {doc_chunks_dir}")

            # Step 3: Embeddings generation
            st.write("Step 3/4: Generating BGE-M3 & SigLIP 2 embeddings...")
            self.run_script("generate_embeddings.py", str(doc_chunks_dir))
            doc_embeddings_dir = self.embeddings_dir / document_id
            if not doc_embeddings_dir.exists():
                raise FileNotFoundError(f"Embedding directory not found: {doc_embeddings_dir}")

            # Step 4: Vector database indexing
            st.write("Step 4/4: Indexing vectors in Qdrant...")
            self.run_script("index_vectors.py", str(doc_embeddings_dir))

            status.update(
                label=f"PDF '{pdf_path.name}' processed and indexed successfully!",
                state="complete",
                expanded=False,
            )

        return {
            "document_id": document_id,
            "filename": pdf_path.name,
            "metadata_path": str(metadata_path),
        }

    def get_indexed_documents(self) -> List[Dict[str, str]]:
        """Scans metadata and raw directories to return currently indexed documents."""
        docs = []
        if self.metadata_dir.exists():
            for meta_file in sorted(self.metadata_dir.glob("*.json")):
                doc_id = meta_file.stem
                raw_pdf = self.raw_dir / f"{doc_id}.pdf"
                filename = raw_pdf.name if raw_pdf.exists() else f"{doc_id}.pdf"
                docs.append({
                    "document_id": doc_id,
                    "filename": filename,
                })
        return docs

    def delete_document(self, document_id: str) -> None:
        """Deletes raw PDF, metadata, chunks, embeddings, and extracted images for a document."""
        import shutil

        # 1. Raw PDF
        raw_pdf = self.raw_dir / f"{document_id}.pdf"
        if raw_pdf.exists():
            raw_pdf.unlink(missing_ok=True)

        # 2. Metadata JSON
        meta_file = self.metadata_dir / f"{document_id}.json"
        if meta_file.exists():
            meta_file.unlink(missing_ok=True)

        # 3. Chunks folder
        chunks_folder = self.chunks_dir / document_id
        if chunks_folder.exists():
            shutil.rmtree(chunks_folder, ignore_errors=True)

        # 4. Embeddings folder
        emb_folder = self.embeddings_dir / document_id
        if emb_folder.exists():
            shutil.rmtree(emb_folder, ignore_errors=True)

        # 5. Extracted images folder
        images_folder = self.root / "data" / "processed" / "images" / document_id
        if images_folder.exists():
            shutil.rmtree(images_folder, ignore_errors=True)

    def clear_all_documents(self) -> None:
        """Removes all raw PDFs, processed artifacts, and resets the Qdrant vector database."""
        import shutil

        # Delete all files in raw_dir (except .gitkeep)
        if self.raw_dir.exists():
            for item in self.raw_dir.iterdir():
                if item.name != ".gitkeep":
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)

        # Clean all subfolders in data/processed
        processed_dir = self.root / "data" / "processed"
        for sub_dir in [
            "chunks", "metadata", "embeddings", "images", "qdrant",
            "tables", "text", "contexts", "image_descriptions"
        ]:
            target = processed_dir / sub_dir
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

        # Clean generated diagrams and responses
        gen_dir = self.root / "data" / "generated"
        for sub_dir in ["diagrams", "responses"]:
            target = gen_dir / sub_dir
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

