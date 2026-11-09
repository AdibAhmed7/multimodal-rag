"""Retrieval Evaluation Script (Phase 10).

Executes automated benchmarks over text and image retrieval pipelines,
calculates Precision@K, Recall@K, and MRR, tracks system resource consumption,
and saves detailed JSON metrics to evaluation/results/.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluation.evaluation_runner import RetrievalEvaluationRunner
from src.evaluation.performance_monitor import PerformanceMonitor, get_gpu_memory
from src.retrieval.retrieval_pipeline import MultimodalRetrievalPipeline
from src.vectorstore.qdrant_manager import QdrantManager


def save_results(results: Dict[str, Any], filename: str) -> Path:
    output_dir = repo_root / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return output_path


def main():
    print("=" * 65)
    print("PHASE 10: MULTIMODAL RETRIEVAL BENCHMARK & EVALUATION")
    print("=" * 65)

    monitor = PerformanceMonitor()
    qdrant_dir = repo_root / "data" / "processed" / "qdrant"
    manager = QdrantManager(database_path=qdrant_dir)

    try:
        print("\nLoading Multimodal Retrieval Pipeline (BGE-M3 & SigLIP 2)...")
        with monitor.measure("pipeline_initialization"):
            pipeline = MultimodalRetrievalPipeline(manager.client)

        runner = RetrievalEvaluationRunner(pipeline)
        text_dataset = repo_root / "evaluation" / "datasets" / "text_queries.json"
        image_dataset = repo_root / "evaluation" / "datasets" / "image_queries.json"

        # 1. Text Retrieval Evaluation
        print("\nRunning Text Retrieval Evaluation (Precision@5, Recall@5, MRR)...")
        with monitor.measure("text_retrieval_benchmark"):
            text_eval = runner.evaluate_text(text_dataset, k=5)

        save_results(text_eval, "text_results.json")
        t_summary = text_eval["summary"]

        print("\n========== TEXT RETRIEVAL ==========")
        print(f"precision_at_k     : {t_summary['precision_at_k']:.4f}")
        print(f"recall_at_k        : {t_summary['recall_at_k']:.4f}")
        print(f"mrr                : {t_summary['mrr']:.4f}")
        print(f"page_precision@k   : {t_summary['page_precision_at_k']:.4f}")
        print(f"page_recall@k      : {t_summary['page_recall_at_k']:.4f}")
        print(f"page_mrr           : {t_summary['page_mrr']:.4f}")

        # 2. Image Retrieval Evaluation
        print("\nRunning Image Retrieval Evaluation (Precision@5, Recall@5, MRR)...")
        with monitor.measure("image_retrieval_benchmark"):
            image_eval = runner.evaluate_images(image_dataset, k=5)

        save_results(image_eval, "image_results.json")
        i_summary = image_eval["summary"]

        print("\n========== IMAGE RETRIEVAL ==========")
        print(f"precision_at_k     : {i_summary['precision_at_k']:.4f}")
        print(f"recall_at_k        : {i_summary['recall_at_k']:.4f}")
        print(f"mrr                : {i_summary['mrr']:.4f}")
        print(f"page_precision@k   : {i_summary['page_precision_at_k']:.4f}")
        print(f"page_recall@k      : {i_summary['page_recall_at_k']:.4f}")
        print(f"page_mrr           : {i_summary['page_mrr']:.4f}")

        # 3. Compile Consolidated Evaluation Summary
        perf_records = monitor.get_results()
        summary_payload = {
            "evaluation_status": "success",
            "text_retrieval": t_summary,
            "image_retrieval": i_summary,
            "performance_metrics": perf_records,
            "gpu_stats": get_gpu_memory(),
        }
        summary_path = save_results(summary_payload, "evaluation_summary.json")

        print("\n========== PERFORMANCE PROFILE ==========")
        for p in perf_records:
            print(f"- {p['operation']}: {p['duration_seconds']}s (RAM RSS: {p['process_memory_mb']} MB)")

        print(f"\nEvaluation completed. Results saved to {summary_path}")

    finally:
        manager.close()


if __name__ == "__main__":
    main()

