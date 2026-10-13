from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    source_file: str
    related_image_ids: list[str] = Field(default_factory=list)


class ImageRecord(BaseModel):
    image_id: str
    document_id: str
    page_number: int = Field(ge=1)
    image_index: int = Field(ge=1)
    image_path: str
    extension: str
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    surrounding_text: str = ""
    related_text_chunk_ids: list[str] = Field(default_factory=list)
