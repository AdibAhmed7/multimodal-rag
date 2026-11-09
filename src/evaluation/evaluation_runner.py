import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


class RetrievalEvaluationRunner:
    """Executes evaluation benchmarks over text and image retrieval pipelines."""

    def __init__(self, retrieval_pipeline):
        self.pipeline = retrieval_pipeline

    @staticmethod
    def load_dataset(dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
        with open(dataset_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def evaluate_text(
        self,
        dataset_path: Union[str, Path],
        k: int = 5,
    ) -> Dict[str, Any]:
        """Evaluates text retrieval precision@K, recall@K, and MRR."""
        dataset = self.load_dataset(dataset_path)
        query_results: List[Dict[str, Any]] = []

        for item in dataset:
            query = item["query"]
            relevant_chunks = item.get("relevant_chunk_ids", [])
            doc_id = item.get("document_id")

            search_res = self.pipeline.search_text(
                query=query,
                limit=k,
                document_id=doc_id,
            )
            retrieved_chunk_ids = [
                r.get("chunk_id") or r.get("id") for r in search_res
            ]
            retrieved_pages = [r.get("page_number") for r in search_res if r.get("page_number")]

            # Compute chunk-level metrics
            p_k = precision_at_k(retrieved_chunk_ids, relevant_chunks, k=k)
            r_k = recall_at_k(retrieved_chunk_ids, relevant_chunks, k=k)
            rr = reciprocal_rank(retrieved_chunk_ids, relevant_chunks)

            # Also compute page-level metrics if relevant_pages provided
            rel_pages = [int(p) for p in item.get("relevant_pages", [])]
            page_p = precision_at_k(retrieved_pages, rel_pages, k=k) if rel_pages else p_k
            page_r = recall_at_k(retrieved_pages, rel_pages, k=k) if rel_pages else r_k
            page_rr = reciprocal_rank(retrieved_pages, rel_pages) if rel_pages else rr

            query_results.append({
                "query_id": item.get("query_id"),
                "query": query,
                "document_id": doc_id,
                "retrieved_ids": retrieved_chunk_ids,
                "relevant_ids": relevant_chunks,
                "retrieved_pages": retrieved_pages,
                "relevant_pages": rel_pages,
                "precision_at_k": round(p_k, 4),
                "recall_at_k": round(r_k, 4),
                "reciprocal_rank": round(rr, 4),
                "page_precision_at_k": round(page_p, 4),
                "page_recall_at_k": round(page_r, 4),
                "page_reciprocal_rank": round(page_rr, 4),
            })

        avg_p = sum(q["precision_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        avg_r = sum(q["recall_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        mrr = mean_reciprocal_rank([q["reciprocal_rank"] for q in query_results])

        avg_page_p = sum(q["page_precision_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        avg_page_r = sum(q["page_recall_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        page_mrr = mean_reciprocal_rank([q["page_reciprocal_rank"] for q in query_results])

        return {
            "k": k,
            "queries_evaluated": len(query_results),
            "summary": {
                "precision_at_k": round(avg_p, 4),
                "recall_at_k": round(avg_r, 4),
                "mrr": round(mrr, 4),
                "page_precision_at_k": round(avg_page_p, 4),
                "page_recall_at_k": round(avg_page_r, 4),
                "page_mrr": round(page_mrr, 4),
            },
            "queries": query_results,
        }

    def evaluate_images(
        self,
        dataset_path: Union[str, Path],
        k: int = 5,
    ) -> Dict[str, Any]:
        """Evaluates image retrieval precision@K, recall@K, and MRR."""
        dataset = self.load_dataset(dataset_path)
        query_results: List[Dict[str, Any]] = []

        for item in dataset:
            query = item["query"]
            relevant_images = item.get("relevant_image_ids", [])
            doc_id = item.get("document_id")

            search_res = self.pipeline.search_images(
                query=query,
                limit=k,
                document_id=doc_id,
            )
            retrieved_img_ids = [
                r.get("image_id") or r.get("id") for r in search_res
            ]
            retrieved_pages = [r.get("page_number") for r in search_res if r.get("page_number")]

            p_k = precision_at_k(retrieved_img_ids, relevant_images, k=k)
            r_k = recall_at_k(retrieved_img_ids, relevant_images, k=k)
            rr = reciprocal_rank(retrieved_img_ids, relevant_images)

            rel_pages = [int(p) for p in item.get("relevant_pages", [])]
            page_p = precision_at_k(retrieved_pages, rel_pages, k=k) if rel_pages else p_k
            page_r = recall_at_k(retrieved_pages, rel_pages, k=k) if rel_pages else r_k
            page_rr = reciprocal_rank(retrieved_pages, rel_pages) if rel_pages else rr

            query_results.append({
                "query_id": item.get("query_id"),
                "query": query,
                "document_id": doc_id,
                "retrieved_ids": retrieved_img_ids,
                "relevant_ids": relevant_images,
                "retrieved_pages": retrieved_pages,
                "relevant_pages": rel_pages,
                "precision_at_k": round(p_k, 4),
                "recall_at_k": round(r_k, 4),
                "reciprocal_rank": round(rr, 4),
                "page_precision_at_k": round(page_p, 4),
                "page_recall_at_k": round(page_r, 4),
                "page_reciprocal_rank": round(page_rr, 4),
            })

        avg_p = sum(q["precision_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        avg_r = sum(q["recall_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        mrr = mean_reciprocal_rank([q["reciprocal_rank"] for q in query_results])

        avg_page_p = sum(q["page_precision_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        avg_page_r = sum(q["page_recall_at_k"] for q in query_results) / len(query_results) if query_results else 0.0
        page_mrr = mean_reciprocal_rank([q["page_reciprocal_rank"] for q in query_results])

        return {
            "k": k,
            "queries_evaluated": len(query_results),
            "summary": {
                "precision_at_k": round(avg_p, 4),
                "recall_at_k": round(avg_r, 4),
                "mrr": round(mrr, 4),
                "page_precision_at_k": round(avg_page_p, 4),
                "page_recall_at_k": round(avg_page_r, 4),
                "page_mrr": round(page_mrr, 4),
            },
            "queries": query_results,
        }
