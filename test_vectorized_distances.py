#!/usr/bin/env python3
"""
Quick test to validate the vectorized distance calculation implementation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import time


def old_implementation(embeddings, labels):
    """Original nested loop implementation"""
    euclidean_distances = []
    cosine_similarities = []
    same_player_pairs = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # Euclidean distance
            euclidean_dist = np.linalg.norm(embeddings[i] - embeddings[j])
            euclidean_distances.append(euclidean_dist)
            
            # Cosine similarity
            sim = F.cosine_similarity(torch.tensor(embeddings[i]).unsqueeze(0), torch.tensor(embeddings[j]).unsqueeze(0))
            cosine_similarities.append(sim.item())
            
            # Same player or not
            is_same_player = labels[i] == labels[j]
            same_player_pairs.append(is_same_player)
    
    return np.array(euclidean_distances), np.array(cosine_similarities), np.array(same_player_pairs)


def new_implementation(embeddings, labels):
    """New vectorized implementation"""
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_array = np.array(labels)
    
    # Vectorized computation
    euclidean_distance_matrix = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T)
    same_player_matrix = (labels_array[:, None] == labels_array[None, :])
    
    # Extract upper triangular part
    n = len(embeddings)
    upper_tri_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    
    euclidean_distances = euclidean_distance_matrix[upper_tri_mask]
    cosine_similarities = cosine_similarity_matrix[upper_tri_mask]
    same_player_pairs = same_player_matrix[upper_tri_mask.numpy()]
    
    return euclidean_distances.numpy(), cosine_similarities.numpy(), same_player_pairs


def test_implementations():
    """Test that both implementations produce the same results"""
    print("Testing vectorized distance calculation implementation...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 50
    embedding_dim = 128
    
    embeddings = np.random.randn(n_samples, embedding_dim)
    labels = ['player_' + str(i % 10) for i in range(n_samples)]  # 10 different players
    
    print(f"Test data: {n_samples} embeddings, {embedding_dim} dimensions, {len(set(labels))} unique players")
    
    # Test old implementation
    print("\nRunning old implementation...")
    start_time = time.time()
    old_dist, old_sim, old_same = old_implementation(embeddings, labels)
    old_time = time.time() - start_time
    print(f"Old implementation time: {old_time:.4f}s")
    
    # Test new implementation
    print("\nRunning new implementation...")
    start_time = time.time()
    new_dist, new_sim, new_same = new_implementation(embeddings, labels)
    new_time = time.time() - start_time
    print(f"New implementation time: {new_time:.4f}s")
    
    print(f"Speedup: {old_time/new_time:.1f}x")
    
    # Compare results
    print("\nComparing results...")
    
    # Check if results are close (allow for small floating point differences)
    dist_close = np.allclose(old_dist, new_dist, rtol=1e-5, atol=1e-8)
    sim_close = np.allclose(old_sim, new_sim, rtol=1e-5, atol=1e-8)
    same_identical = np.array_equal(old_same, new_same)
    
    print(f"Euclidean distances match: {dist_close}")
    print(f"Cosine similarities match: {sim_close}")
    print(f"Same player labels match: {same_identical}")
    
    # Calculate actual differences for debugging
    dist_max_diff = np.max(np.abs(old_dist - new_dist))
    sim_max_diff = np.max(np.abs(old_sim - new_sim))
    
    print(f"Max distance difference: {dist_max_diff:.10f}")
    print(f"Max similarity difference: {sim_max_diff:.10f}")
    
    # Consider it a success if differences are within machine precision
    dist_ok = dist_max_diff < 1e-5  # Relaxed tolerance for practical purposes
    sim_ok = sim_max_diff < 1e-5    # Relaxed tolerance for practical purposes
    
    if dist_ok and sim_ok and same_identical:
        print("\n✅ SUCCESS: All results match within acceptable tolerance!")
    else:
        print("\n❌ ERROR: Results don't match!")
        
        if not dist_ok:
            print(f"Distance differences too large: max={dist_max_diff:.10f}")
        if not sim_ok:
            print(f"Similarity differences too large: max={sim_max_diff:.10f}")
        if not same_identical:
            print(f"Label differences: {np.sum(old_same != new_same)} mismatches")
    
    # Test statistics computation
    print("\nTesting statistics computation...")
    
    same_mask = new_same
    different_mask = ~new_same
    
    if np.any(same_mask):
        avg_dist_same = float(new_dist[same_mask].mean())
        avg_sim_same = float(new_sim[same_mask].mean())
    else:
        avg_dist_same = avg_sim_same = float('nan')
        
    if np.any(different_mask):
        avg_dist_different = float(new_dist[different_mask].mean())
        avg_sim_different = float(new_sim[different_mask].mean())
    else:
        avg_dist_different = avg_sim_different = float('nan')
    
    print(f"Same player pairs: {np.sum(same_mask)}")
    print(f"Different player pairs: {np.sum(different_mask)}")
    print(f"Avg distance (same): {avg_dist_same:.4f}")
    print(f"Avg distance (different): {avg_dist_different:.4f}")
    print(f"Distance separation: {avg_dist_different - avg_dist_same:.4f}")
    print(f"Avg similarity (same): {avg_sim_same:.4f}")
    print(f"Avg similarity (different): {avg_sim_different:.4f}")
    print(f"Similarity separation: {avg_sim_same - avg_sim_different:.4f}")


if __name__ == "__main__":
    test_implementations()
