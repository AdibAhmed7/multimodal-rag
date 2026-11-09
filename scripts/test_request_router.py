import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.generation.request_router import RequestRouter


def main():
    router = RequestRouter()
    test_queries = [
        "Explain the main concept of this PDF",
        "Show me the diagram from the PDF",
        "Generate a new neural network illustration",
        "Generate a new diagram and show the original PDF diagram",
    ]

    print("=" * 60)
    print("TESTING REQUEST ROUTER")
    print("=" * 60)

    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery  : {query}")
        print(f"Intent : {result['intent']}")
        print(f"Flags  : retrieve_pdf={result['retrieve_pdf_images']}, generate_img={result['generate_new_image']}, answer_qa={result['answer_question']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

