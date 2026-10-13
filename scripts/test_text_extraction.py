from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_extractor import TextExtractor

pdf_path = project_root / "data" / "raw" / "pdfs" / "example.pdf"
extractor = TextExtractor()

with PDFLoader(pdf_path) as pdf:
    pages = extractor.extract_document(pdf)
    for page in pages:
        print("=" * 80)
        print(f"PAGE {page['page_number']}")
        print("=" * 80)
        print(page["text"][:1000])
