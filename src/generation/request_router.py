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
        "generate an illustration",
    ]

    DIAGRAM_KEYWORDS = [
        "flowchart",
        "mindmap",
        "mind map",
        "sequence diagram",
        "class diagram",
        "er diagram",
        "erd",
        "state diagram",
        "timeline",
        "architecture diagram",
        "block diagram",
        "process diagram",
        "generate a diagram",
        "create a diagram",
        "draw a diagram",
        "draw diagram",
        "make a diagram",
        "generate diagram",
        "create diagram",
        "draw a flowchart",
        "create a flowchart",
        "generate a flowchart",
        "draw a sequence diagram",
        "draw a mind map",
        "draw a mindmap",
        "mermaid",
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
        has_image_query = any(
            k in normalized
            for k in [
                "image",
                "diagram",
                "figure",
                "illustration",
                "picture",
                "chart",
                "flowchart",
                "photo",
            ]
        )
        wants_retrieval = any(keyword in normalized for keyword in self.RETRIEVE_KEYWORDS) or (
            has_image_query
            and any(
                k in normalized
                for k in [
                    "pdf",
                    "document",
                    "show",
                    "find",
                    "display",
                    "original",
                    "view",
                    "look at",
                    "from",
                ]
            )
            and not any(d_verb in normalized for d_verb in ["draw", "create", "generate", "make"])
        )
        wants_diagram = any(keyword in normalized for keyword in self.DIAGRAM_KEYWORDS) and not wants_retrieval
        wants_generation = any(keyword in normalized for keyword in self.GENERATE_KEYWORDS) and not wants_diagram
        is_explanation = any(keyword in normalized for keyword in self.EXPLAIN_KEYWORDS)

        generate_diagram = False

        if wants_diagram:
            generate_diagram = True
            retrieve_pdf_images = False
            generate_new_image = False
            if is_explanation:
                intent = "answer_and_diagram"
                answer_question = True
            else:
                intent = "generate_diagram"
                answer_question = False
        elif wants_generation and wants_retrieval:
            intent = "answer_and_generate"
            retrieve_pdf_images = True
            generate_new_image = True
            answer_question = True
        elif wants_generation:
            if is_explanation:
                intent = "answer_and_generate"
                retrieve_pdf_images = False
                generate_new_image = True
                answer_question = True
            else:
                intent = "generate_image"
                retrieve_pdf_images = False
                generate_new_image = True
                answer_question = False
        elif wants_retrieval:
            if is_explanation:
                intent = "answer"
                retrieve_pdf_images = True
                generate_new_image = False
                answer_question = True
            else:
                intent = "retrieve_pdf_images"
                retrieve_pdf_images = True
                generate_new_image = False
                answer_question = False
        else:
            # Normal text question: provide answer only, no images unless specifically asked
            intent = "answer"
            retrieve_pdf_images = False
            generate_new_image = False
            answer_question = True

        return {
            "intent": intent,
            "retrieve_pdf_images": retrieve_pdf_images,
            "generate_new_image": generate_new_image,
            "generate_diagram": generate_diagram,
            "answer_question": answer_question,
        }

