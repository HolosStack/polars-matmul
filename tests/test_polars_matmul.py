"""Tests for polars-matmul"""

import pytest
import polars as pl
import numpy as np

import polars_matmul as pmm


class TestSimilarityJoin:
    """Tests for the similarity_join function"""
    
    def test_basic_cosine(self):
        """Test basic cosine similarity join"""
        queries = pl.DataFrame({
            "query_id": [0, 1],
            "embedding": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        })
        
        corpus = pl.DataFrame({
            "corpus_id": [0, 1, 2],
            "embedding": [
                [1.0, 0.0, 0.0],  # Exact match for query 0
                [0.0, 1.0, 0.0],  # Exact match for query 1
                [0.0, 0.0, 1.0],  # Orthogonal to both
            ],
            "label": ["a", "b", "c"],
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="cosine",
        )
        
        # Should have 4 rows (2 queries × 2 top-k)
        assert len(result) == 4
        
        # Check columns
        assert "query_id" in result.columns
        assert "corpus_id" in result.columns
        assert "label" in result.columns
        assert "_score" in result.columns
        
        # Embedding columns should be excluded
        assert "embedding" not in result.columns
        
        # Check that query 0's top match is corpus 0 (cosine = 1.0)
        query0_results = result.filter(pl.col("query_id") == 0)
        top_match = query0_results.sort("_score", descending=True).row(0)
        assert top_match[1] == 0  # corpus_id
        assert abs(top_match[3] - 1.0) < 1e-6  # score
    
    def test_dot_product(self):
        """Test dot product metric"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[2.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [0, 1],
            "embedding": [[1.0, 0.0], [3.0, 0.0]],
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="dot",
        )
        
        # Top match should be [3, 0] with dot product 6.0
        top = result.sort("_score", descending=True).row(0)
        assert top[1] == 1  # corpus id
        assert abs(top[2] - 6.0) < 1e-6  # score (2*3 = 6)
    
    def test_euclidean(self):
        """Test euclidean distance metric"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[0.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [0, 1],
            "embedding": [[3.0, 4.0], [1.0, 0.0]],  # distances: 5, 1
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=2,
            metric="euclidean",
        )
        
        # For euclidean, lower is better - so [1, 0] should be first
        top = result.sort("_score").row(0)
        assert top[1] == 1  # corpus id with distance 1
        assert abs(top[2] - 1.0) < 1e-6
    
    def test_lazyframe(self):
        """Test that LazyFrame input returns LazyFrame output"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        }).lazy()
        
        corpus = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        }).lazy()
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=1,
        )
        
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert len(collected) == 1
    
    def test_k_larger_than_corpus(self):
        """Test when k > corpus size"""
        queries = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
        })
        
        corpus = pl.DataFrame({
            "id": [0, 1],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        })
        
        result = pmm.similarity_join(
            left=queries,
            right=corpus,
            left_on="embedding",
            right_on="embedding",
            k=10,  # Much larger than corpus
        )
        
        # Should return all 2 corpus items
        assert len(result) == 2
    
    def test_suffix_handling(self):
        """Test column name suffix for conflicts"""
        df = pl.DataFrame({
            "id": [0],
            "embedding": [[1.0, 0.0]],
            "value": [100],
        })
        
        result = pmm.similarity_join(
            left=df,
            right=df,
            left_on="embedding",
            right_on="embedding",
            k=1,
            suffix="_corpus",
        )
        
        # Should have both 'value' and 'value_corpus'
        assert "value" in result.columns
        assert "value_corpus" in result.columns


class TestMatmul:
    """Tests for matmul function"""
    
    def test_basic(self):
        """Test basic matrix multiplication"""
        left = pl.Series("l", [[1.0, 2.0], [3.0, 4.0]])
        right = pl.Series("r", [[1.0, 0.0], [0.0, 1.0]])
        
        result = pmm.matmul(left, right)
        
        assert result.len() == 2
        # [1, 2] @ [[1, 0], [0, 1]]^T = [1*1 + 2*0, 1*0 + 2*1] = [1, 2]
        assert result[0].to_list() == pytest.approx([1.0, 2.0])
        # [3, 4] @ [[1, 0], [0, 1]]^T = [3*1 + 4*0, 3*0 + 4*1] = [3, 4]
        assert result[1].to_list() == pytest.approx([3.0, 4.0])
    
    def test_against_numpy(self):
        """Verify matmul matches NumPy results"""
        np.random.seed(42)
        left_np = np.random.randn(10, 32)
        right_np = np.random.randn(20, 32)
        
        left = pl.Series("l", left_np.tolist())
        right = pl.Series("r", right_np.tolist())
        
        result = pmm.matmul(left, right)
        expected = left_np @ right_np.T
        
        for i in range(10):
            actual = result[i].to_list()
            np.testing.assert_allclose(actual, expected[i], rtol=1e-5)


class TestNumpyEquivalence:
    """Tests verifying equivalence with NumPy implementations"""
    
    def test_cosine_similarity_via_join(self):
        """Test cosine similarity matches NumPy implementation"""
        np.random.seed(42)
        query = np.random.randn(5, 16)
        corpus = np.random.randn(20, 16)
        
        # Normalize for cosine similarity
        query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
        corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        expected = query_norm @ corpus_norm.T
        
        query_df = pl.DataFrame({
            "id": range(5),
            "embedding": query.tolist(),
        })
        corpus_df = pl.DataFrame({
            "id": range(20),
            "embedding": corpus.tolist(),
        })
        
        # Get all matches (k=20)
        result = pmm.similarity_join(
            left=query_df,
            right=corpus_df,
            left_on="embedding",
            right_on="embedding",
            k=20,
            metric="cosine",
        )
        
        # Check that for each query, the scores match the expected values
        for i in range(5):
            query_results = result.filter(pl.col("id") == i).sort("id_right")
            actual_scores = query_results["_score"].to_list()
            expected_scores = sorted(expected[i].tolist(), reverse=True)
            actual_sorted = sorted(actual_scores, reverse=True)
            np.testing.assert_allclose(actual_sorted, expected_scores, rtol=1e-5)
