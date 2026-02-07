"""
polars-matmul: BLAS-accelerated similarity joins for Polars

This package provides fast similarity search operations on embedding columns
using BLAS-accelerated matrix multiplication.

Example:
    >>> import polars as pl
    >>> import polars_matmul as pmm
    >>> 
    >>> query_df = pl.DataFrame({
    ...     "query_id": [0, 1],
    ...     "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ... })
    >>> corpus_df = pl.DataFrame({
    ...     "corpus_id": [0, 1, 2],
    ...     "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ...     "label": ["a", "b", "c"],
    ... })
    >>> 
    >>> result = pmm.similarity_join(
    ...     left=query_df,
    ...     right=corpus_df,
    ...     left_on="embedding",
    ...     right_on="embedding",
    ...     k=2,
    ...     metric="cosine",
    ... )
"""

from polars_matmul._polars_matmul import (
    _similarity_join_eager,
    _matmul,
)
import polars as pl
from typing import Literal, Union

__version__ = "0.1.0"
__all__ = ["similarity_join", "matmul"]


Metric = Literal["cosine", "dot", "euclidean"]


def similarity_join(
    left: Union[pl.DataFrame, pl.LazyFrame],
    right: Union[pl.DataFrame, pl.LazyFrame],
    left_on: str,
    right_on: str,
    k: int,
    metric: Metric = "cosine",
    suffix: str = "_right",
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Perform a similarity join between two DataFrames based on embedding columns.
    
    This function finds the top-k most similar rows from `right` for each row in `left`,
    using BLAS-accelerated matrix multiplication for high performance.
    
    Args:
        left: The query DataFrame or LazyFrame
        right: The corpus DataFrame or LazyFrame to search
        left_on: Name of the embedding column in `left`
        right_on: Name of the embedding column in `right`
        k: Number of top matches to return per query row
        metric: Similarity metric to use:
            - "cosine": Cosine similarity (default, best for normalized embeddings)
            - "dot": Raw dot product
            - "euclidean": Euclidean distance (lower = more similar)
        suffix: Suffix to append to column names from `right` that conflict with `left`
    
    Returns:
        A DataFrame/LazyFrame with all columns from `left`, plus:
        - All columns from `right` (k rows per left row)
        - A `_score` column with the similarity/distance value
        
    Example:
        >>> result = pmm.similarity_join(
        ...     left=queries,
        ...     right=corpus,
        ...     left_on="embedding",
        ...     right_on="embedding",
        ...     k=10,
        ...     metric="cosine",
        ... )
    """
    # Handle LazyFrame by collecting
    is_lazy = isinstance(left, pl.LazyFrame)
    left_df = left.collect() if is_lazy else left
    right_df = right.collect() if isinstance(right, pl.LazyFrame) else right
    
    # Validate inputs
    if left_on not in left_df.columns:
        raise ValueError(f"Column '{left_on}' not found in left DataFrame")
    if right_on not in right_df.columns:
        raise ValueError(f"Column '{right_on}' not found in right DataFrame")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(right_df):
        k = len(right_df)
    
    # Call the Rust implementation
    result = _similarity_join_eager(
        left_df,
        right_df,
        left_on,
        right_on,
        k,
        metric,
        suffix,
    )
    
    return result.lazy() if is_lazy else result


def matmul(
    left: pl.Series,
    right: pl.Series,
) -> pl.Series:
    """
    Compute the full matrix multiplication between two embedding series.
    
    Returns a Series of List[f64] where each element contains the dot products
    of one left vector with all right vectors.
    
    Args:
        left: Series of embedding vectors (List or Array type)
        right: Series of embedding vectors (List or Array type)
    
    Returns:
        Series of List[f64] with shape (len(left), len(right))
        
    Example:
        >>> similarities = pmm.matmul(queries["embedding"], corpus["embedding"])
        >>> # similarities[i] contains dot products of query i with all corpus vectors
    """
    return _matmul(left, right)
