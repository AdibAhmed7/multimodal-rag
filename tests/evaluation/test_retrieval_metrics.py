from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


def test_precision():
    result = precision_at_k(
        ["A", "X", "B", "Y", "Z"],
        ["A", "B", "C"],
        k=5,
    )
    assert result == 0.4


def test_recall():
    result = recall_at_k(
        ["A", "X", "B", "Y", "Z"],
        ["A", "B", "C"],
        k=5,
    )
    assert abs(result - (2 / 3)) < 0.001


def test_reciprocal_rank():
    result = reciprocal_rank(
        ["X", "B", "A"],
        ["A", "B"],
    )
    assert result == 0.5

    result_top1 = reciprocal_rank(
        ["A", "X", "B"],
        ["A", "B"],
    )
    assert result_top1 == 1.0


def test_mean_reciprocal_rank():
    scores = [1.0, 0.5, 0.25]
    mrr = mean_reciprocal_rank(scores)
    assert abs(mrr - (1.75 / 3)) < 0.001

