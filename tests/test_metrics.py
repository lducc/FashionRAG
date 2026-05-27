from eval.metrics import mrr, recall_at_k


def test_recall_at_k_counts_relevant_results():
    results = [{"id": "a"}, {"id": "x"}, {"id": "b"}, {"id": "y"}]

    assert recall_at_k(results, ["a", "b", "c"], k=3) == 2 / 3


def test_recall_at_k_handles_empty_relevant_ids():
    assert recall_at_k([{"id": "a"}], [], k=5) == 0.0


def test_mrr_uses_first_relevant_rank():
    results = [{"id": "x"}, {"id": "b"}, {"id": "a"}]

    assert mrr(results, ["a", "b"]) == 0.5


def test_mrr_returns_zero_when_no_match():
    assert mrr([{"id": "x"}], ["a"]) == 0.0
