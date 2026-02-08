//! Top-k selection implementation

use ndarray::Array2;

/// Select top-k indices and scores from similarity/distance matrix (f64)
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

/// Select top-k indices and scores from similarity/distance matrix (f32)
pub fn select_topk_with_scores_f32(
    matrix: &Array2<f32>,
    k: usize,
    higher_is_better: bool,
) -> (Array2<usize>, Array2<f32>) {
    let n_queries = matrix.nrows();
    let mut indices = Array2::zeros((n_queries, k));
    let mut scores = Array2::zeros((n_queries, k));
    
    for (i, row) in matrix.outer_iter().enumerate() {
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        
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
    fn test_topk_higher_is_better_f64() {
        let matrix = array![
            [0.1f64, 0.9, 0.5],
            [0.8, 0.2, 0.6]
        ];
        
        let (indices, _scores) = select_topk_with_scores(&matrix, 2, true);
        
        assert_eq!(indices[[0, 0]], 1);
        assert_eq!(indices[[0, 1]], 2);
        assert_eq!(indices[[1, 0]], 0);
        assert_eq!(indices[[1, 1]], 2);
    }
    
    #[test]
    fn test_topk_higher_is_better_f32() {
        let matrix = array![
            [0.1f32, 0.9, 0.5],
            [0.8, 0.2, 0.6]
        ];
        
        let (indices, _scores) = select_topk_with_scores_f32(&matrix, 2, true);
        
        assert_eq!(indices[[0, 0]], 1);
        assert_eq!(indices[[0, 1]], 2);
        assert_eq!(indices[[1, 0]], 0);
        assert_eq!(indices[[1, 1]], 2);
    }
    
    #[test]
    fn test_topk_lower_is_better_f64() {
        let matrix = array![
            [0.1f64, 0.9, 0.5],
            [0.8, 0.2, 0.6]
        ];
        
        let (indices, _scores) = select_topk_with_scores(&matrix, 2, false);
        
        assert_eq!(indices[[0, 0]], 0);
        assert_eq!(indices[[0, 1]], 2);
        assert_eq!(indices[[1, 0]], 1);
        assert_eq!(indices[[1, 1]], 2);
    }
}
