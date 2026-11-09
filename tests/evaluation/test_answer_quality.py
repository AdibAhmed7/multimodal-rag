from src.evaluation.citation_validator import (
    extract_page_citations,
    validate_citations,
)


def test_extract_page_citations():
    text = "The model uses self-attention [Page 2] and feedforward layers [Page 5]."
    pages = extract_page_citations(text)
    assert pages == [2, 5]


def test_validate_citations_all_valid():
    answer = "A neural network contains multiple layers. [Page 5]"
    result = validate_citations(answer, available_pages=[3, 4, 5, 6])
    assert result["has_citations"] is True
    assert result["citation_valid"] is True
    assert result["invalid_pages"] == []


def test_validate_citations_with_invalid():
    answer = "A neural network contains multiple layers. [Page 5] The output layer generates predictions. [Page 8]"
    result = validate_citations(answer, available_pages=[5, 6, 7])
    assert result["cited_pages"] == [5, 8]
    assert result["invalid_pages"] == [8]
    assert result["citation_valid"] is False
    assert result["has_citations"] is True


def test_validate_citations_empty():
    answer = "No page citations provided in this sentence."
    result = validate_citations(answer, available_pages=[1, 2])
    assert result["has_citations"] is False
    assert result["citation_valid"] is True
    assert result["cited_pages"] == []

