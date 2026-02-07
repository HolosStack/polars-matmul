//! Core matrix multiplication and similarity join implementation

use ndarray::Array2;
use polars::prelude::*;
use crate::metrics::{Metric, compute_similarity_matrix};
use crate::topk::select_topk_with_scores;

/// Convert a Polars Series of List/Array to an ndarray matrix
pub fn series_to_matrix(series: &Series) -> PolarsResult<Array2<f64>> {
    // Handle both List and Array types
    let series = series.cast(&DataType::List(Box::new(DataType::Float64)))?;
    let ca = series.list()?;
    
    let n_rows = ca.len();
    if n_rows == 0 {
        return Err(PolarsError::ComputeError("Empty series".into()));
    }
    
    // Get dimension from first non-null element
    let dim = ca.get_as_series(0)
        .ok_or_else(|| PolarsError::ComputeError("First element is null".into()))?
        .len();
    
    if dim == 0 {
        return Err(PolarsError::ComputeError("Zero-dimensional vectors".into()));
    }
    
    // Allocate matrix
    let mut matrix = Array2::zeros((n_rows, dim));
    
    // Fill matrix row by row
    for i in 0..n_rows {
        if let Some(inner) = ca.get_as_series(i) {
            let values = inner.f64()?;
            for (j, val) in values.iter().enumerate() {
                matrix[[i, j]] = val.unwrap_or(0.0);
            }
        }
        // Null rows remain as zeros
    }
    
    Ok(matrix)
}

/// Compute full matrix multiplication between two Series
pub fn matmul_impl(left: &Series, right: &Series) -> PolarsResult<Series> {
    let left_matrix = series_to_matrix(left)?;
    let right_matrix = series_to_matrix(right)?;
    
    // Compute dot products: left @ right^T
    let result = left_matrix.dot(&right_matrix.t());
    
    // Convert back to Polars Series of List[f64]
    let lists: Vec<Series> = result
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let values: Vec<f64> = row.to_vec();
            Series::new(format!("{}", i).into(), values)
        })
        .collect();
    
    Series::new("matmul".into(), lists).cast(&DataType::List(Box::new(DataType::Float64)))
}

/// Main similarity join implementation
pub fn similarity_join_impl(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &str,
    right_on: &str,
    k: usize,
    metric_str: &str,
    suffix: &str,
) -> PolarsResult<DataFrame> {
    let metric = Metric::from_str(metric_str)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    
    // Extract embedding columns
    let left_embeddings = left.column(left_on)?;
    let right_embeddings = right.column(right_on)?;
    
    // Convert to matrices
    let query_matrix = series_to_matrix(left_embeddings.as_materialized_series())?;
    let corpus_matrix = series_to_matrix(right_embeddings.as_materialized_series())?;
    
    let k = k.min(corpus_matrix.nrows());
    
    // Compute similarity matrix using BLAS
    let similarity = compute_similarity_matrix(&query_matrix, &corpus_matrix, metric);
    
    // Select top-k
    let (topk_indices, topk_scores) = select_topk_with_scores(&similarity, k, metric.higher_is_better());
    
    // Build result DataFrame
    // For each left row, we have k right rows
    // Total rows = n_queries * k
    
    // Collect left column names for conflict detection
    let left_col_names: Vec<PlSmallStr> = left.get_columns()
        .iter()
        .map(|c| c.name().clone())
        .collect();
    
    // Expand left DataFrame: repeat each row k times
    let mut all_cols: Vec<Column> = Vec::new();
    for col in left.get_columns() {
        let col_name = col.name();
        // Skip the embedding column in output (too large, not useful)
        if col_name.as_str() == left_on {
            continue;
        }
        
        // Repeat each value k times
        let expanded = repeat_each(col.as_materialized_series(), k)?;
        all_cols.push(expanded.into_column());
    }
    
    // Gather right rows based on indices
    let flat_indices: Vec<u32> = topk_indices.iter().map(|&x| x as u32).collect();
    let indices_series = Series::new("idx".into(), flat_indices);
    let idx_ca = indices_series.idx()?;
    
    for col in right.get_columns() {
        let col_name = col.name().clone();
        // Skip the embedding column
        if col_name.as_str() == right_on {
            continue;
        }
        
        // Gather rows by index
        // SAFETY: indices are valid (from topk selection)
        let gathered = unsafe { col.as_materialized_series().take_unchecked(idx_ca) };
        
        // Rename if conflicts with left
        let new_name = if left_col_names.contains(&col_name) {
            PlSmallStr::from(format!("{}{}", col_name, suffix))
        } else {
            col_name
        };
        
        all_cols.push(gathered.with_name(new_name).into_column());
    }
    
    // Add score column
    let scores: Vec<f64> = topk_scores.iter().copied().collect();
    let score_series = Series::new("_score".into(), scores);
    all_cols.push(score_series.into_column());
    
    DataFrame::new(all_cols)
}

/// Repeat each element of a Series k times
fn repeat_each(series: &Series, k: usize) -> PolarsResult<Series> {
    let n = series.len();
    let indices: Vec<u32> = (0..n as u32)
        .flat_map(|i| std::iter::repeat(i).take(k))
        .collect();
    
    let idx_series = Series::new("idx".into(), indices);
    let idx_ca = idx_series.idx()?;
    
    // SAFETY: indices are valid
    Ok(unsafe { series.take_unchecked(idx_ca) })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_series_to_matrix() {
        let s = Series::new("test".into(), vec![
            Series::new("".into(), vec![1.0f64, 2.0, 3.0]),
            Series::new("".into(), vec![4.0f64, 5.0, 6.0]),
        ]);
        
        let matrix = series_to_matrix(&s).unwrap();
        
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3);
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((matrix[[1, 2]] - 6.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_repeat_each() {
        let s = Series::new("test".into(), vec![1i32, 2, 3]);
        let repeated = repeat_each(&s, 2).unwrap();
        
        assert_eq!(repeated.len(), 6);
        let values: Vec<i32> = repeated.i32().unwrap().into_no_null_iter().collect();
        assert_eq!(values, vec![1, 1, 2, 2, 3, 3]);
    }
}
