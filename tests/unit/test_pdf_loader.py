from pathlib import Path
import pytest
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_extractor import TextExtractor
from src.ingestion.image_extractor import ImageExtractor
from src.ingestion.page_parser import PageParser
from src.ingestion.metadata_builder import MetadataBuilder

EXAMPLE_PDF = Path("data/raw/pdfs/example.pdf")


def test_pdf_loader_valid():
    with PDFLoader(EXAMPLE_PDF) as loader:
        assert loader.page_count == 3
        assert "title" in loader.metadata
        page = loader.get_page(0)
        assert page is not None


def test_pdf_loader_invalid_file():
    with pytest.raises(FileNotFoundError):
        PDFLoader("non_existent_file.pdf")


def test_pdf_loader_invalid_extension(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("hello")
    with pytest.raises(ValueError):
        PDFLoader(txt_file)


def test_pdf_loader_out_of_bounds():
    with PDFLoader(EXAMPLE_PDF) as loader:
        with pytest.raises(IndexError):
            loader.get_page(99)
        with pytest.raises(IndexError):
            loader.get_page(-1)


def test_text_extractor():
    extractor = TextExtractor()
    with PDFLoader(EXAMPLE_PDF) as loader:
        pages = extractor.extract_document(loader)
        assert len(pages) == 3
        assert pages[0]["page_number"] == 1
        assert "Transformers" in pages[0]["text"]


def test_image_extractor(tmp_path):
    extractor = ImageExtractor()
    with PDFLoader(EXAMPLE_PDF) as loader:
        page2 = loader.get_page(1)
        images = extractor.extract_page_images(page2, page_number=2, output_dir=tmp_path)
        assert len(images) == 1
        assert images[0]["image_id"] == "page_002_img_001"
        assert Path(images[0]["path"]).exists()


def test_page_parser(tmp_path):
    parser = PageParser()
    with PDFLoader(EXAMPLE_PDF) as loader:
        page_dict = parser.parse_page(loader, page_number=2, image_output_dir=tmp_path)
        assert page_dict["page_number"] == 2
        assert "Transformer Architecture" in page_dict["text"]
        assert len(page_dict["images"]) == 1


def test_metadata_builder(tmp_path):
    parser = PageParser()
    builder = MetadataBuilder()
    with PDFLoader(EXAMPLE_PDF) as loader:
        pages = parser.parse_document(loader, image_output_dir=tmp_path / "images")
        metadata = builder.build(loader, pages)
        out_json = tmp_path / "meta.json"
        builder.save(metadata, out_json)
        assert out_json.exists()
        assert metadata["document_id"] == "example"
        assert metadata["page_count"] == 3
