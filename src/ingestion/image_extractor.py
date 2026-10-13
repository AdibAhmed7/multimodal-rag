from pathlib import Path


class ImageExtractor:
    """Extract embedded images from PDF pages."""

    def extract_page_images(
        self,
        page,
        page_number: int,
        output_dir: str | Path,
    ) -> list[dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images = page.get_images(full=True)
        extracted_images = []

        for image_index, image_info in enumerate(images, start=1):
            xref = image_info[0]
            image_data = page.parent.extract_image(xref)
            image_bytes = image_data["image"]
            image_ext = image_data["ext"]
            filename = f"page_{page_number:03d}_img_{image_index:03d}.{image_ext}"
            image_path = output_dir / filename

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_id = f"page_{page_number:03d}_img_{image_index:03d}"
            extracted_images.append({
                "image_id": image_id,
                "page_number": page_number,
                "path": str(image_path.as_posix()),
                "extension": image_ext,
                "xref": xref,
                "width": image_data.get("width"),
                "height": image_data.get("height"),
            })

        return extracted_images
