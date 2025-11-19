use crate::matrix::Matrix;

/// Trait for loss functions
pub trait Loss {
    /// Compute the loss value
    fn loss(&self, predicted: &Matrix, target: &Matrix) -> f64;

    /// Compute the derivative of loss with respect to predictions
    fn derivative(&self, predicted: &Matrix, target: &Matrix) -> Matrix;
}

/// Mean Squared Error: measures average squared difference
/// Formula: (1/n) * Σ(predicted - target)²
pub struct MSE;

impl Loss for MSE {
    /// Compute MSE loss
    fn loss(&self, predicted: &Matrix, target: &Matrix) -> f64 {
        debug_assert_eq!(predicted.rows, target.rows);
        debug_assert_eq!(predicted.cols, target.cols);

        // Difference between predicted and target
        // Squaring allows us to penalize larger errors more while not overly punishing small errors
        // however this does mean it's quite sensitive to outliers
        let mut sum = 0.0;
        for i in 0..predicted.rows {
            for j in 0..predicted.cols {
                let diff = predicted.data[i][j] - target.data[i][j];
                sum += diff * diff;
            }
        }

        // Divide by total number of elements (rows * cols) to get mean
        sum / (predicted.rows * predicted.cols) as f64
    }

    /// Derivative of MSE with respect to predictions
    /// Formula: 2 * (predicted - target) / n
    fn derivative(&self, predicted: &Matrix, target: &Matrix) -> Matrix {
        debug_assert_eq!(predicted.rows, target.rows);
        debug_assert_eq!(predicted.cols, target.cols);

        // Difference between predicted and target
        let diff = predicted.subtract(&target);

        // Scale by 2.0 / (rows * cols)
        // This gives the average gradient per element
        let scale = 2.0 / (predicted.rows * predicted.cols) as f64;
        diff.scale(scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss() { 
        let mse = MSE;
        let predicted = super::Matrix::from_vec(vec![
            vec![0.5, 0.2, 0.1],
            vec![0.4, 0.6, 0.9],
        ]);

        let target = super::Matrix::from_vec(vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ]);

        let loss_value = mse.loss(&predicted, &target);
        assert!((loss_value - 0.13833333333333334).abs() < f64::EPSILON);
    }

    #[test]
    fn derivative() {
        let mse = MSE;
        let predicted = super::Matrix::from_vec(vec![
            vec![0.5, 0.2, 0.1],
            vec![0.4, 0.6, 0.9],
        ]);

        let target = super::Matrix::from_vec(vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ]);

        let derivative = mse.derivative(&predicted, &target);
        let expected = super::Matrix::from_vec(vec![
            vec![0.16666666666666666, 0.066666666666666666, 0.033333333333333333],
            vec![-0.19999999999999998, -0.13333333333333333, -0.033333333333333326],
        ]);

        for i in 0..derivative.rows {
            for j in 0..derivative.cols {
                assert!((derivative.data[i][j] - expected.data[i][j]).abs() < f64::EPSILON);
            }
        }
    }
}