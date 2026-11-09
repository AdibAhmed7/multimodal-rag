from src.retrieval.result_fusion import reciprocal_rank_fusion


def test_rrf_combines_results():
    text_results = [
        {
            "id": "1",
            "result_type": "text",
            "content": "Neural network explanation",
        },
        {
            "id": "2",
            "result_type": "text",
            "content": "Deep learning explanation",
        },
    ]
    image_results = [
        {
            "id": "3",
            "result_type": "image",
            "image_path": "image1.png",
        },
        {
            "id": "4",
            "result_type": "image",
            "image_path": "image2.png",
        },
    ]
    results = reciprocal_rank_fusion([
        text_results,
        image_results,
    ])
    assert len(results) == 4
    assert all("rrf_score" in r for r in results)
    # Validate sorted descending by rrf_score
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
