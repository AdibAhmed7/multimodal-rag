from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.image_extractor import ImageExtractor

pdf_path = project_root / "data" / "raw" / "pdfs" / "example.pdf"
output_dir = project_root / "data" / "processed" / "images" / "example"
extractor = ImageExtractor()

with PDFLoader(pdf_path) as pdf:
    total_images = 0
    for page_number in range(1, pdf.page_count + 1):
        page = pdf.get_page(page_number - 1)
        images = extractor.extract_page_images(
            page=page,
            page_number=page_number,
            output_dir=output_dir,
        )
        total_images += len(images)
        if images:
            print(f"Page {page_number}: {len(images)} image(s)")
            for img in images:
                print(f"  - Extracted: {img['image_id']} -> {img['path']}")
    print(f"\nTotal images extracted: {total_images}")
