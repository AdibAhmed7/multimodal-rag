from typing import Any, Dict, List


class PromptBuilder:
    """Constructs evidence-grounded prompts enforcing strict citation and faithfulness rules."""

    SYSTEM_PROMPT = """You are a multimodal PDF question-answering assistant.
Your task is to answer the user's question using the
provided PDF evidence, including text and images.

STRICT RULES:
1. Use the supplied PDF evidence as your primary source.
2. Do not invent facts, values, labels or diagram components.
3. If the evidence is insufficient, explicitly say so.
4. You may use your general knowledge to explain terminology,
   but distinguish it from information found in the PDF.
5. When discussing an image, inspect the original image.
6. Do not claim that an image contains something that is
   not visibly supported.
7. Cite the source page using [Page N].
8. If multiple pages support a statement, cite each relevant page.
9. Do not fabricate page numbers or source references.
10. Answer in clear, well-organized language.

If the user requests a diagram or image:
- Identify the relevant retrieved image.
- Explain what it shows.
- Include its image ID and page number in the response
  so the application can display the original image.

If the PDF does not contain sufficient evidence, say:
"The provided document does not contain enough information
to answer this question reliably."
"""

    def build(self, query: str, pages: List[Dict[str, Any]]) -> str:
        """Format retrieved pages and user question into structured prompt text."""
        sections: List[str] = [
            f"User Question: {query}\n",
            "Supplied PDF Evidence:\n",
        ]

        if not pages:
            sections.append("No relevant document sections found.\n")
        else:
            for page in pages:
                doc_id = page.get("document_id", "unknown")
                page_num = page.get("page_number", 0)
                sections.append(f"--- Document: {doc_id} | Page: {page_num} ---")

                # Text chunks on page
                chunks = page.get("text_chunks", [])
                for chunk in chunks:
                    chunk_id = chunk.get("chunk_id", "c")
                    content = chunk.get("content", "").strip()
                    sections.append(f"[Chunk {chunk_id}]: {content}")

                # Images on page
                images = page.get("images", [])
                for img in images:
                    img_id = img.get("image_id", "img")
                    desc = img.get("description")
                    surr = img.get("surrounding_text")
                    img_info = f"[Image {img_id}]"
                    if desc:
                        img_info += f" Description: {desc}"
                    elif surr:
                        img_info += f" Context: {surr}"
                    sections.append(img_info)

                sections.append("")

        sections.append(
            "Instructions:\n"
            "Synthesize a clear and comprehensive answer to the user's question. "
            "Rely strictly on the provided evidence above and any attached images. "
            "Every key statement or claim MUST cite its source page in brackets, like [Page N]."
        )

        return "\n".join(sections)

