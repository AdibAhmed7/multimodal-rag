"""Response Display Manager for Streamlit UI.

Renders generated text, original PDF figures, and AI-synthesized illustrations
with distinct badges, citations, and layout controls.
"""

from pathlib import Path
from typing import Any, Dict
import streamlit as st

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
        """Renders text answers, retrieved images, and generated images."""
        answer = response.get("answer")
        if answer:
            st.markdown(answer)

        # 1. Display original PDF images
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

        # 2. Display newly generated AI images
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

        # 3. Display sources and citations expander
        sources = response.get("sources", [])
        stats = response.get("statistics", {})
        if sources or stats:
            with st.expander("📚 Source Evidence & Retrieval Stats", expanded=False):
                if stats:
                    st.caption(
                        f"**Retrieved Images:** {stats.get('retrieved_image_count', 0)} | "
                        f"**Generated Images:** {stats.get('generated_image_count', 0)} | "
                        f"**Evidence Sources:** {stats.get('source_count', len(sources))}"
                    )
                for s in sources:
                    s_type = s.get("type", "source").upper()
                    page_num = s.get("page_number", "N/A")
                    s_id = s.get("id", s.get("image_id", ""))
                    st.markdown(f"- **[{s_type}]** Page {page_num} — `{s_id}`")

