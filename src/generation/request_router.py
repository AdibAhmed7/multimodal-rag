"""Request router for multimodal RAG queries.

Distinguishes between:
- Text answer generation ("answer")
- Document image retrieval ("retrieve_pdf_images")
- New AI image generation ("generate_image")
- Hybrid retrieval and generation ("answer_and_generate")
"""

from typing import Any, Dict


class RequestRouter:
    """Classifies user queries into specific retrieval and generation intents."""

    GENERATE_KEYWORDS = [
        "generate an image",
        "generate image",
        "create an image",
        "create image",
        "draw an image",
        "draw me",
        "make an image",
        "make me an image",
        "illustrate",
        "illustration",
        "generate a picture",
        "create a picture",
        "generate a new",
        "generate a diagram",
        "create a new diagram",
        "generate an illustration",
    ]

    RETRIEVE_KEYWORDS = [
        "show me the diagram",
        "show the diagram",
        "find the diagram",
        "show me the image",
        "show the image",
        "show the figure",
        "find the image",
        "diagram from the pdf",
        "diagram from the document",
        "image from the pdf",
        "figure from the pdf",
        "existing diagram",
        "original diagram",
        "original pdf diagram",
    ]

    EXPLAIN_KEYWORDS = [
        "explain",
        "describe",
        "what is",
        "what are",
        "how does",
        "how do",
        "why",
        "tell me about",
        "summarize",
        "summary",
        "concept",
    ]

    def route(self, query: str) -> Dict[str, Any]:
        """Routes the input query to determine execution intent.

        Args:
            query: User's input question or instruction.

        Returns:
            Dictionary containing intent and boolean flags for pipeline execution.
        """
        normalized = query.lower().strip()

        wants_generation = any(keyword in normalized for keyword in self.GENERATE_KEYWORDS)
        wants_retrieval = any(keyword in normalized for keyword in self.RETRIEVE_KEYWORDS) or (
            "original" in normalized and ("diagram" in normalized or "image" in normalized or "pdf" in normalized)
        )
        is_explanation = any(keyword in normalized for keyword in self.EXPLAIN_KEYWORDS)

        if wants_generation and wants_retrieval:
            intent = "answer_and_generate"
            retrieve_pdf_images = True
            generate_new_image = True
            answer_question = True
        elif wants_generation:
            intent = "generate_image"
            retrieve_pdf_images = False
            generate_new_image = True
            answer_question = False
        elif wants_retrieval and not is_explanation:
            intent = "retrieve_pdf_images"
            retrieve_pdf_images = True
            generate_new_image = False
            answer_question = False
        else:
            intent = "answer"
            retrieve_pdf_images = True
            generate_new_image = False
            answer_question = True

        return {
            "intent": intent,
            "retrieve_pdf_images": retrieve_pdf_images,
            "generate_new_image": generate_new_image,
            "answer_question": answer_question,
        }
