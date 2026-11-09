import re
from typing import Any, Dict, List, Sequence, Union


def extract_page_citations(answer: str) -> List[int]:
    """Extract all cited page numbers matching [Page N] format."""
    pattern = r"\[Page\s+(\d+)\]"
    matches = re.findall(pattern, answer, flags=re.IGNORECASE)
    return [int(page) for page in matches]


def validate_citations(
    answer: str,
    available_pages: Sequence[Union[int, str]],
) -> Dict[str, Any]:
    """Verify that cited pages were present in the retrieved evidence."""
    cited_pages = extract_page_citations(answer)
    available = {int(p) for p in available_pages}
    invalid_pages = [page for page in cited_pages if page not in available]
    return {
        "cited_pages": cited_pages,
        "invalid_pages": invalid_pages,
        "citation_valid": len(invalid_pages) == 0,
        "has_citations": len(cited_pages) > 0,
    }
