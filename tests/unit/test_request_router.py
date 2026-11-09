from src.generation.request_router import RequestRouter


def test_normal_question():
    router = RequestRouter()
    result = router.route("Explain the main concept of this PDF")
    assert result["intent"] == "answer"


def test_image_generation():
    router = RequestRouter()
    result = router.route("Generate an image of a neural network")
    assert result["intent"] == "generate_image"


def test_image_retrieval():
    router = RequestRouter()
    result = router.route("Show me the diagram from the PDF")
    assert result["intent"] == "retrieve_pdf_images"

