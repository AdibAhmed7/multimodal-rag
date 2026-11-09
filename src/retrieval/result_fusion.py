from typing import Any, Dict, List, Sequence


def reciprocal_rank_fusion(
    result_lists: Sequence[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Combine ranked result lists using Reciprocal Rank Fusion (RRF).

    Combines ranking positions from separate heterogeneous search spaces (e.g.
    BGE-M3 text and SigLIP 2 vision) without assuming identical calibration of scores.

    Args:
        result_lists: Sequence of ordered result lists.
        k: Smoothing constant (standard default is 60).

    Returns:
        Consolidated result list sorted in descending order of rrf_score.
    """
    fused: Dict[tuple, Dict[str, Any]] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            result_id = (
                result.get("result_type"),
                result.get("id"),
            )
            if result_id not in fused:
                fused[result_id] = {
                    **result,
                    "rrf_score": 0.0,
                }
            fused[result_id]["rrf_score"] += 1.0 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
