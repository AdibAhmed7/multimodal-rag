"""Response Display Manager for Streamlit UI.

Renders generated text, original PDF figures, AI-synthesized illustrations,
and AI-generated Mermaid.js diagrams with distinct badges, citations, and layout controls.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import streamlit as st
import streamlit.components.v1 as components

from src.diagram_generator.diagram_utils import build_mermaid_html

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DisplayManager:
    """Formats and renders multimodal responses in the Streamlit interface."""

    @staticmethod
    def resolve_image_path(path_str: str) -> Path:
        """Resolves image paths across relative and absolute formats."""
        p = Path(path_str)
        if not p.is_absolute() and not p.exists():
            candidate = PROJECT_ROOT / path_str
            if candidate.exists():
                return candidate
        return p

    @classmethod
    def display_response(cls, response: Dict[str, Any]) -> None:
        """Renders text answers, Mermaid diagrams, retrieved images, and generated images."""
        answer = response.get("answer")
        if answer:
            st.markdown(answer)

        # 1. Display Mermaid Diagram (if generated)
        diagram = response.get("diagram")
        if diagram and diagram.get("mermaid_code"):
            mermaid_code = diagram["mermaid_code"]
            diag_type = diagram.get("diagram_type", "diagram")
            st.markdown(f"##### 📊 Technical Diagram (`{diag_type}`)")
            diagram_html = build_mermaid_html(mermaid_code, height=540)
            components.html(diagram_html, height=590, scrolling=True)

            dcol1, dcol2 = st.columns([1, 2])
            with dcol1:
                st.download_button(
                    label="📥 Download Mermaid (.mmd)",
                    data=mermaid_code,
                    file_name=f"{diag_type}_diagram.mmd",
                    mime="text/plain",
                    key=f"dl_mmd_{abs(hash(mermaid_code)) % 10000000}",
                    use_container_width=True,
                )
            with dcol2:
                st.caption("💡 Pan/Zoom, SVG & PNG download controls are built directly into the diagram viewer above.")

            with st.expander("📝 View Mermaid Code", expanded=False):
                st.code(mermaid_code, language="mermaid")

        # 2. Display original PDF images
        retrieved_images = response.get("retrieved_images", [])
        if retrieved_images:
            st.markdown("##### 📄 Images Retrieved from PDF")
            num_cols = min(max(len(retrieved_images), 1), 2)
            cols = st.columns(num_cols)
            for idx, img in enumerate(retrieved_images):
                img_path = cls.resolve_image_path(img.get("image_path", ""))
                if img_path.exists():
                    target_col = cols[idx % num_cols]
                    score_str = f" | Match score: {img.get('score'):.4f}" if img.get("score") is not None else ""
                    caption = f"Page {img.get('page_number')} (Original PDF Figure{score_str})"
                    target_col.image(str(img_path), caption=caption, use_container_width=True)

        # 3. Display newly generated AI images
        generated_images = response.get("generated_images", [])
        if generated_images:
            st.markdown("##### 🎨 AI-Generated Concept Illustration")
            for img in generated_images:
                img_path = cls.resolve_image_path(img.get("image_path", ""))
                if img_path.exists():
                    caption = (
                        f"AI-Generated Image | Model: {img.get('model', 'SD-Turbo')} "
                        f"(Steps: {img.get('steps', 2)}, Seed: {img.get('seed', 42)})"
                    )
                    st.image(str(img_path), caption=caption, use_container_width=True)

        # 4. Display sources and citations expander
        sources = response.get("sources", [])
        stats = response.get("statistics", {})
        if sources or stats:
            with st.expander("📚 Source Evidence & Retrieval Stats", expanded=False):
                if stats:
                    st.caption(
                        f"**Retrieved Images:** {stats.get('retrieved_image_count', 0)} | "
                        f"**Generated Images:** {stats.get('generated_image_count', 0)} | "
                        f"**Diagram:** {'Yes' if stats.get('has_diagram') else 'None'} | "
                        f"**Evidence Sources:** {stats.get('source_count', len(sources))}"
                    )
                for s in sources:
                    s_type = s.get("type", "source").upper()
                    page_num = s.get("page_number", "N/A")
                    s_id = s.get("id", s.get("image_id", ""))
                    doc_id = s.get("document_id", "")
                    content_prev = s.get("content", "")
                    doc_str = f" | Doc: `{doc_id}`" if doc_id else ""
                    prev_str = f" — *\"{content_prev}\"*" if content_prev else ""
                    st.markdown(f"- **[{s_type}]** Page {page_num}{doc_str} `{s_id}`{prev_str}")
