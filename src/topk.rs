//! Top-k selection implementation

use ndarray::Array2;
use polars::prelude::*;
use crate::metrics::{Metric, compute_similarity_matrix};
use crate::matmul::series_to_matrix;

/// Find top-k indices for each query vector
pub fn topk_indices_impl(
    query: &Series,
    corpus: &Series,
    k: usize,
    metric_str: &str,
) -> PolarsResult<Series> {
    let metric = Metric::from_str(metric_str)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    
    let query_matrix = series_to_matrix(query)?;
    let corpus_matrix = series_to_matrix(corpus)?;
    
    let k = k.min(corpus_matrix.nrows());
    
    let similarity = compute_similarity_matrix(&query_matrix, &corpus_matrix, metric);
    let indices = select_topk_indices(&similarity, k, metric.higher_is_better());
    
    // Convert to Polars Series of List[u32]
    let lists: Vec<Series> = indices
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let values: Vec<u32> = row.iter().map(|&x| x as u32).collect();
            Series::new(format!("{}", i).into(), values)
        })
        .collect();
    
    Series::new("topk_indices".into(), lists).cast(&DataType::List(Box::new(DataType::UInt32)))
}

/// Get top-k scores for each query vector
pub fn topk_scores_impl(
    query: &Series,
    corpus: &Series,
    k: usize,
    metric_str: &str,
) -> PolarsResult<Series> {
    let metric = Metric::from_str(metric_str)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    
    let query_matrix = series_to_matrix(query)?;
    let corpus_matrix = series_to_matrix(corpus)?;
    
    let k = k.min(corpus_matrix.nrows());
    
    let similarity = compute_similarity_matrix(&query_matrix, &corpus_matrix, metric);
    let (_, scores) = select_topk_with_scores(&similarity, k, metric.higher_is_better());
    
    // Convert to Polars Series of List[f64]
    let lists: Vec<Series> = scores
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let values: Vec<f64> = row.to_vec();
            Series::new(format!("{}", i).into(), values)
        })
        .collect();
    
    Series::new("topk_scores".into(), lists).cast(&DataType::List(Box::new(DataType::Float64)))
}

/// Select top-k indices from similarity/distance matrix
/// 
/// Uses partial sort (similar to np.argpartition) for efficiency when k << n
pub fn select_topk_indices(
    matrix: &Array2<f64>,
    k: usize,
    higher_is_better: bool,
) -> Array2<usize> {
    let n_queries = matrix.nrows();
    let mut result = Array2::zeros((n_queries, k));
    
    for (i, row) in matrix.outer_iter().enumerate() {
        let mut indexed: Vec<(usize, f64)> = row.iter().copied().enumerate().collect();
        
        // Partial sort: only sort enough to get top-k
        if higher_is_better {
            indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        for (j, (idx, _)) in indexed.iter().enumerate() {
            result[[i, j]] = *idx;
        }
    }
    
    result
}

/// Select top-k indices and scores from similarity/distance matrix
pub fn select_topk_with_scores(
    matrix: &Array2<f64>,
    k: usize,
    higher_is_better: bool,
) -> (Array2<usize>, Array2<f64>) {
    let n_queries = matrix.nrows();
    let mut indices = Array2::zeros((n_queries, k));
    let mut scores = Array2::zeros((n_queries, k));
    
    for (i, row) in matrix.outer_iter().enumerate() {
        let mut indexed: Vec<(usize, f64)> = row.iter().copied().enumerate().collect();
        
        if higher_is_better {
            indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        for (j, (idx, score)) in indexed.iter().enumerate() {
            indices[[i, j]] = *idx;
            scores[[i, j]] = *score;
        }
    }
    
    (indices, scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_topk_higher_is_better() {
        let matrix = array![
            [0.1, 0.9, 0.5],
            [0.8, 0.2, 0.6]
        ];
        
        let indices = select_topk_indices(&matrix, 2, true);
        
        // First row: 0.9 (idx 1), 0.5 (idx 2)
        assert_eq!(indices[[0, 0]], 1);
        assert_eq!(indices[[0, 1]], 2);
        
        // Second row: 0.8 (idx 0), 0.6 (idx 2)
        assert_eq!(indices[[1, 0]], 0);
        assert_eq!(indices[[1, 1]], 2);
    }
    
    #[test]
    fn test_topk_lower_is_better() {
        let matrix = array![
            [0.1, 0.9, 0.5],
            [0.8, 0.2, 0.6]
        ];
        
        let indices = select_topk_indices(&matrix, 2, false);
        
        // First row: 0.1 (idx 0), 0.5 (idx 2)
        assert_eq!(indices[[0, 0]], 0);
        assert_eq!(indices[[0, 1]], 2);
        
        // Second row: 0.2 (idx 1), 0.6 (idx 2)
        assert_eq!(indices[[1, 0]], 1);
        assert_eq!(indices[[1, 1]], 2);
    }
}
