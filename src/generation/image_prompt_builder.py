"""Prompt builder for AI image generation.

Expands raw user queries and optional PDF context into detailed visual instructions
optimized for educational and technical illustrations.
"""

from typing import Optional


class ImagePromptBuilder:
    """Builds structured, high-quality prompts for diffusion models."""

    def build(self, user_prompt: str, context: Optional[str] = None) -> str:
        """Constructs an enhanced prompt for image generation.

        Args:
            user_prompt: The user's input request or concept to illustrate.
            context: Optional contextual summary or notes from the PDF.

        Returns:
            Enhanced prompt string with visual requirements and hierarchy.
        """
        prompt = f"""Create a high-quality educational illustration.
Main subject: {user_prompt}
Visual requirements:
- Clear composition.
- Visually understandable.
- Good spacing between components.
- Clean and professional appearance.
- Appropriate visual hierarchy.
- High contrast.
- Avoid unnecessary decorative elements."""

        if context:
            prompt += (
                f"\nAdditional information from the PDF:\n{context}\n"
                "Use this information to guide the illustration.\n"
                "Do not introduce unsupported technical claims."
            )

        return prompt.strip()

