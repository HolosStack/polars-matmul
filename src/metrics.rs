//! Similarity metrics implementation

use ndarray::{Array1, Array2, Axis};

/// Supported similarity/distance metrics
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    /// Cosine similarity: dot(a, b) / (||a|| * ||b||)
    Cosine,
    /// Raw dot product
    Dot,
    /// Euclidean distance (L2)
    Euclidean,
}

impl Metric {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Metric::Cosine),
            "dot" => Ok(Metric::Dot),
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            _ => Err(format!("Unknown metric: '{}'. Supported: cosine, dot, euclidean", s)),
        }
    }
    
    /// Returns true if higher scores are better (similarity), false if lower is better (distance)
    pub fn higher_is_better(&self) -> bool {
        match self {
            Metric::Cosine | Metric::Dot => true,
            Metric::Euclidean => false,
        }
    }
}

/// Compute similarity/distance matrix between query and corpus matrices
/// 
/// Returns a matrix of shape (n_queries, n_corpus) with similarity/distance values
pub fn compute_similarity_matrix(
    query: &Array2<f64>,
    corpus: &Array2<f64>,
    metric: Metric,
) -> Array2<f64> {
    match metric {
        Metric::Dot => {
            // Simple matrix multiplication: Q @ C^T
            query.dot(&corpus.t())
        }
        Metric::Cosine => {
            // Normalize vectors then dot product
            let query_norms = compute_norms(query);
            let corpus_norms = compute_norms(corpus);
            
            let mut result = query.dot(&corpus.t());
            
            // Divide by norms
            for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                let q_norm = query_norms[i];
                if q_norm > 1e-10 {
                    for (j, val) in row.iter_mut().enumerate() {
                        let c_norm = corpus_norms[j];
                        if c_norm > 1e-10 {
                            *val /= q_norm * c_norm;
                        } else {
                            *val = 0.0;
                        }
                    }
                } else {
                    row.fill(0.0);
                }
            }
            result
        }
        Metric::Euclidean => {
            // ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            // Then take sqrt
            let query_sq_norms = compute_squared_norms(query);
            let corpus_sq_norms = compute_squared_norms(corpus);
            
            let dot_products = query.dot(&corpus.t());
            
            let n_queries = query.nrows();
            let n_corpus = corpus.nrows();
            let mut result = Array2::zeros((n_queries, n_corpus));
            
            for i in 0..n_queries {
                for j in 0..n_corpus {
                    let sq_dist = query_sq_norms[i] + corpus_sq_norms[j] - 2.0 * dot_products[[i, j]];
                    // Clamp to avoid negative values from floating point errors
                    result[[i, j]] = sq_dist.max(0.0).sqrt();
                }
            }
            result
        }
    }
}

/// Compute L2 norms for each row
fn compute_norms(matrix: &Array2<f64>) -> Array1<f64> {
    matrix.map_axis(Axis(1), |row| {
        row.dot(&row).sqrt()
    })
}

/// Compute squared L2 norms for each row
fn compute_squared_norms(matrix: &Array2<f64>) -> Array1<f64> {
    matrix.map_axis(Axis(1), |row| {
        row.dot(&row)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_dot_product() {
        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let corpus = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        
        let result = compute_similarity_matrix(&query, &corpus, Metric::Dot);
        
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let corpus = array![[2.0, 0.0], [0.0, 3.0]]; // Different magnitudes
        
        let result = compute_similarity_matrix(&query, &corpus, Metric::Cosine);
        
        // Cosine should be 1.0 for parallel vectors regardless of magnitude
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let query = array![[0.0, 0.0]];
        let corpus = array![[3.0, 4.0]]; // Distance should be 5
        
        let result = compute_similarity_matrix(&query, &corpus, Metric::Euclidean);
        
        assert!((result[[0, 0]] - 5.0).abs() < 1e-10);
    }
}
