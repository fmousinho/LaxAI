import numpy as np
import torch
import pytest

from src.train.evaluator import ModelEvaluator


def make_dummy_evaluator():
    # simple dummy model; not used by the tests that call internal methods directly
    model = torch.nn.Identity()
    device = torch.device("cpu")
    return ModelEvaluator(model=model, device=device)


def test_compute_pairwise_and_distance_and_classification():
    evalr = make_dummy_evaluator()

    # Create 4 embeddings forming two tight clusters (labels 0 and 1)
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=np.float32)

    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    # Force small batch size to exercise batching logic
    sims, dists, labels_eq = evalr._compute_pairwise_batches(embeddings, labels, batch_size=2, compute_distances=True)

    # Basic sanity checks
    assert sims.numel() > 0
    assert dists.numel() > 0
    assert labels_eq.numel() == sims.numel()

    # Evaluate distances: same-player distances should be smaller than different-player distances
    dist_metrics = evalr._evaluate_distances(sims, dists, labels_eq)
    assert dist_metrics["same_player_pairs_count"] > 0
    assert dist_metrics["different_player_pairs_count"] > 0
    assert dist_metrics["avg_distance_same_player"] < dist_metrics["avg_distance_different_player"]
    assert dist_metrics["similarity_separation"] > 0

    # Classification should find a threshold that separates the two clusters well
    cls_metrics = evalr._evaluate_classification(sims, labels_eq)
    # With clearly separable clusters we expect near-perfect metrics
    assert cls_metrics["f1_score"] == pytest.approx(1.0, rel=1e-3)
    assert cls_metrics["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert cls_metrics["roc_auc"] == pytest.approx(1.0, rel=1e-3)


def test_compute_average_precision_and_ranking():
    evalr = make_dummy_evaluator()

    # Test average precision helper
    relevant = np.array([0, 1, 0, 1])
    ap = evalr._compute_average_precision(relevant)
    # Expected AP for hits at positions [1,3] is mean([1/2, 2/4]) == 0.5
    assert ap == pytest.approx(0.5, rel=1e-6)

    # Ranking evaluation: reuse the small clustered embeddings from above
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=np.float32)

    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    rank_metrics = evalr._evaluate_ranking(embeddings, labels)
    # For the symmetric clusters above, mean average precision is expected (observed 0.75)
    assert rank_metrics["mean_average_precision"] == pytest.approx(0.75, rel=1e-6)
