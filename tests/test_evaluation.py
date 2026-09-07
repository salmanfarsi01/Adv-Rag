from ocr_pipeline.evaluation import EvaluationQuery, evaluate, precision_at_k


def test_precision_at_k_counts_relevant_hits():
    queries = [
        EvaluationQuery("invoice total", ("a",)),
        EvaluationQuery("vendor", ("b",)),
    ]

    def search(query, k):
        return [{"metadata": {"chunk_id": "a" if query == "invoice total" else "x"}}]

    results = evaluate(queries, search, 3)
    assert precision_at_k(results) == 0.5
    assert results[0].hit is True
    assert results[1].hit is False
