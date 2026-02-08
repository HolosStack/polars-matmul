"""Benchmark polars-matmul against NumPy with correctness verification"""

import time
import numpy as np
import polars as pl
import polars_matmul as pmm


def numpy_topk_cosine(query: np.ndarray, corpus: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference implementation for top-k cosine similarity.
    
    Returns (indices, scores) for top-k matches per query.
    """
    # Normalize for cosine similarity
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    
    # Matrix multiplication
    similarities = np.dot(query_norm, corpus_norm.T)
    
    # Top-k selection
    partitioned = np.argpartition(-similarities, k, axis=1)[:, :k]
    rows = np.arange(len(query))[:, None]
    top_k_similarities = similarities[rows, partitioned]
    sorted_within_k = np.argsort(-top_k_similarities, axis=1)
    
    indices = partitioned[rows, sorted_within_k]
    scores = top_k_similarities[rows, sorted_within_k]
    
    return indices, scores


def benchmark_numpy(query: np.ndarray, corpus: np.ndarray, k: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Benchmark NumPy top-k similarity search. Returns (time, indices, scores)."""
    start = time.perf_counter()
    indices, scores = numpy_topk_cosine(query, corpus, k)
    elapsed = time.perf_counter() - start
    return elapsed, indices, scores


def benchmark_polars_matmul(query_df: pl.DataFrame, corpus_df: pl.DataFrame, k: int) -> tuple[float, pl.DataFrame]:
    """Benchmark polars-matmul similarity join. Returns (time, result_df)."""
    start = time.perf_counter()
    
    result = pmm.similarity_join(
        left=query_df,
        right=corpus_df,
        left_on="embedding",
        right_on="embedding",
        k=k,
        metric="cosine",
    )
    
    elapsed = time.perf_counter() - start
    return elapsed, result


def verify_correctness(
    numpy_indices: np.ndarray,
    numpy_scores: np.ndarray,
    pmm_result: pl.DataFrame,
    n_queries: int,
    k: int,
    rtol: float = 1e-5,
) -> bool:
    """Verify that polars-matmul results match NumPy reference."""
    for i in range(n_queries):
        # Get polars-matmul results for query i
        pmm_query = pmm_result.filter(pl.col("query_id") == i).sort("_score", descending=True)
        pmm_indices = pmm_query["corpus_id"].to_list()
        pmm_scores = pmm_query["_score"].to_list()
        
        # Compare scores (indices might differ for ties, but scores should match)
        np_scores_sorted = sorted(numpy_scores[i].tolist(), reverse=True)
        pmm_scores_sorted = sorted(pmm_scores, reverse=True)
        
        if not np.allclose(np_scores_sorted, pmm_scores_sorted, rtol=rtol):
            print(f"  ❌ Score mismatch for query {i}")
            print(f"     NumPy:  {np_scores_sorted[:5]}...")
            print(f"     PMM:    {pmm_scores_sorted[:5]}...")
            return False
    
    return True


def run_benchmark(n_queries: int, n_corpus: int, dim: int, k: int, n_warmup: int = 2, n_runs: int = 5):
    """Run benchmark with given parameters"""
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_queries} queries × {n_corpus} corpus × {dim}d, k={k}")
    print(f"{'='*60}")
    
    # Generate random data
    np.random.seed(42)
    query_np = np.random.randn(n_queries, dim).astype(np.float64)
    corpus_np = np.random.randn(n_corpus, dim).astype(np.float64)
    
    # Convert to Polars
    query_df = pl.DataFrame({
        "query_id": range(n_queries),
        "embedding": query_np.tolist(),
    })
    
    corpus_df = pl.DataFrame({
        "corpus_id": range(n_corpus),
        "embedding": corpus_np.tolist(),
    })
    
    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        benchmark_numpy(query_np, corpus_np, k)
        benchmark_polars_matmul(query_df, corpus_df, k)
    
    # Benchmark NumPy
    numpy_times = []
    numpy_indices = numpy_scores = None
    for _ in range(n_runs):
        elapsed, numpy_indices, numpy_scores = benchmark_numpy(query_np, corpus_np, k)
        numpy_times.append(elapsed)
    numpy_mean = np.mean(numpy_times) * 1000  # ms
    numpy_std = np.std(numpy_times) * 1000
    
    # Benchmark polars-matmul
    pmm_times = []
    pmm_result = None
    for _ in range(n_runs):
        elapsed, pmm_result = benchmark_polars_matmul(query_df, corpus_df, k)
        pmm_times.append(elapsed)
    pmm_mean = np.mean(pmm_times) * 1000  # ms
    pmm_std = np.std(pmm_times) * 1000
    
    # Verify correctness
    print("Verifying correctness...")
    is_correct = verify_correctness(numpy_indices, numpy_scores, pmm_result, n_queries, k)
    
    print(f"\nResults ({n_runs} runs):")
    print(f"  NumPy:         {numpy_mean:8.2f}ms ± {numpy_std:.2f}ms")
    print(f"  polars-matmul: {pmm_mean:8.2f}ms ± {pmm_std:.2f}ms")
    print(f"  Ratio:         {pmm_mean/numpy_mean:.2f}x")
    print(f"  Correctness:   {'✅ PASSED' if is_correct else '❌ FAILED'}")
    
    if not is_correct:
        raise AssertionError("Correctness check failed!")
    
    return numpy_mean, pmm_mean


if __name__ == "__main__":
    print("polars-matmul Benchmark")
    print("=" * 60)
    
    # Small benchmark
    run_benchmark(100, 2000, 100, 10)
    
    # Medium benchmark
    run_benchmark(100, 2000, 1000, 10)
    
    # Large benchmark  
    run_benchmark(1000, 10000, 100, 10)
    
    # Very large (if you have time)
    # run_benchmark(1000, 50000, 100, 10)
    
    print("\n" + "=" * 60)
    print("✅ All benchmarks passed!")
