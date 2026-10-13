from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import PDFLoader

pdf_path = project_root / "data" / "raw" / "pdfs" / "example.pdf"

with PDFLoader(pdf_path) as pdf:
    print("Pages:", pdf.page_count)
    print("Metadata:")
    for key, value in pdf.metadata.items():
        print(f"  {key}: {value}")
