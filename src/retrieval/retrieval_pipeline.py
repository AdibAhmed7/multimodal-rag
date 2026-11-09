from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient

from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.image_retriever import ImageRetriever
from src.retrieval.result_fusion import reciprocal_rank_fusion


class MultimodalRetrievalPipeline:
    """End-to-end multimodal retrieval engine combining text and image retrieval with RRF."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        text_model_name: str = "BAAI/bge-m3",
        image_model_name: str = "google/siglip2-base-patch16-224",
    ):
        print("Initializing retrieval pipeline...")
        self.query_encoder = QueryEncoder(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
        )
        self.text_retriever = TextRetriever(qdrant_client)
        self.image_retriever = ImageRetriever(qdrant_client, self.query_encoder)
        print("Retrieval pipeline ready.")

    def search_text(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant text chunks using BGE-M3 query vector."""
        vector = self.query_encoder.encode_text_for_text_search(query)
        return self.text_retriever.search(
            query_vector=vector,
            limit=limit,
            document_id=document_id,
        )

    def search_images(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant image records using SigLIP 2 text query vector."""
        return self.image_retriever.search_by_text(
            query=query,
            limit=limit,
            document_id=document_id,
        )

    def search_by_image(
        self,
        image_input: Union[str, Path, Image.Image],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve visually similar images using SigLIP 2 vision vector."""
        return self.image_retriever.search_by_image(
            image_input=image_input,
            limit=limit,
            document_id=document_id,
        )

    def search_multimodal(
        self,
        query: str,
        text_limit: int = 5,
        image_limit: int = 5,
        document_id: Optional[str] = None,
        k: int = 60,
    ) -> Dict[str, Any]:
        """Perform multimodal search across text and image collections fused with RRF."""
        text_results = self.search_text(
            query,
            limit=text_limit,
            document_id=document_id,
        )
        image_results = self.search_images(
            query,
            limit=image_limit,
            document_id=document_id,
        )
        fused = reciprocal_rank_fusion([text_results, image_results], k=k)

        return {
            "query": query,
            "text_results": text_results,
            "image_results": image_results,
            "fused_results": fused,
        }
