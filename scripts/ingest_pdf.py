import argparse
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_extractor import TextExtractor
from src.ingestion.image_extractor import ImageExtractor
from src.ingestion.page_parser import PageParser
from src.ingestion.metadata_builder import MetadataBuilder


def ingest_pdf(pdf_path: str | Path, verbose: bool = True) -> dict:
    """Ingest a PDF file, extracting text and images, and save structured metadata."""
    pdf_path = Path(pdf_path)

    # If the provided path is relative and does not exist in current directory,
    # resolve it relative to the project root
    if not pdf_path.is_absolute() and not pdf_path.exists():
        candidate_path = project_root / pdf_path
        if candidate_path.exists():
            pdf_path = candidate_path

    document_id = pdf_path.stem

    image_output_dir = project_root / "data" / "processed" / "images" / document_id
    metadata_output_path = project_root / "data" / "processed" / "metadata" / f"{document_id}.json"

    text_extractor = TextExtractor()
    image_extractor = ImageExtractor()
    page_parser = PageParser(
        text_extractor=text_extractor,
        image_extractor=image_extractor,
    )
    metadata_builder = MetadataBuilder()

    if verbose:
        print(f"\n--- Ingesting PDF: {pdf_path} ---")

    with PDFLoader(pdf_path) as pdf:
        pages = []
        total_images = 0

        for page_number in range(1, pdf.page_count + 1):
            page_data = page_parser.parse_page(
                pdf_loader=pdf,
                page_number=page_number,
                image_output_dir=image_output_dir,
            )
            pages.append(page_data)
            num_images = len(page_data["images"])
            total_images += num_images
            if verbose:
                print(f"Parsed Page {page_number}/{pdf.page_count}: {len(page_data['text'])} chars, {num_images} image(s)")

        metadata = metadata_builder.build(pdf_loader=pdf, pages=pages)
        metadata_builder.save(metadata=metadata, output_path=metadata_output_path)

    if verbose:
        print(f"\nSuccessfully ingested '{pdf_path.name}'!")
        print(f"  Total Pages: {len(pages)}")
        print(f"  Total Images Extracted: {total_images}")
        print(f"  Images Saved To: {image_output_dir}")
        print(f"  Metadata Saved To: {metadata_output_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF document for Multimodal RAG.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="data/raw/pdfs/example.pdf",
        help="Path to the PDF file (default: data/raw/pdfs/example.pdf)",
    )
    args = parser.parse_args()

    ingest_pdf(args.pdf_path)
