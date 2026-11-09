from typing import List, Sequence, Set, Union


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate Precision@K: the fraction of retrieved items in the top-K that are relevant."""
    if k <= 0:
        return 0.0
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for item in retrieved if item in relevant)
    return hits / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate Recall@K: the fraction of relevant items that are retrieved in the top-K."""
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    hits = len(retrieved.intersection(relevant))
    return hits / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant item."""
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across a list of query reciprocal ranks."""
    if not reciprocal_ranks:
        return 0.0
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))
