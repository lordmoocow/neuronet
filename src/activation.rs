use crate::matrix::Matrix;

pub trait Activation {
    fn activate(&self, x: &Matrix) -> Matrix;
    fn derivative(&self, x: &Matrix) -> Matrix;
    fn name(&self) -> &str;
}

pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, v: &Matrix) -> Matrix {
        v.map(|x| x.max(0.0))
    }

    fn derivative(&self, v: &Matrix) -> Matrix {
        v.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn name(&self) -> &str {
        "relu"
    }
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, v: &Matrix) -> Matrix {
        v.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(&self, v: &Matrix) -> Matrix {
        v.map(|x| x * (1.0 - x))
    }

    fn name(&self) -> &str {
        "sigmoid"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sigmoid_activation() {
        use crate::{activation::Sigmoid, matrix::Matrix, activation::Activation};

        let sigmoid = Sigmoid;
        let input = Matrix::from_vec(vec![vec![-2.0, 0.0, 2.0]]);
        let activated = sigmoid.activate(&input);
        let expected = Matrix::from_vec(vec![vec![0.11920292, 0.5, 0.88079708]]);

        for (a, e) in activated.data[0].iter().zip(expected.data[0].iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn sigmoid_derivative() {
        use crate::{activation::Sigmoid, matrix::Matrix, activation::Activation};

        let sigmoid = Sigmoid;
        let input = Matrix::from_vec(vec![vec![0.11920292, 0.5, 0.88079708]]);
        let derivative = sigmoid.derivative(&input);
        let expected = Matrix::from_vec(vec![vec![0.10499359, 0.25, 0.10499359]]);

        for (d, e) in derivative.data[0].iter().zip(expected.data[0].iter()) {
            assert!((d - e).abs() < 1e-6);
        }
    }

    #[test]
    fn relu_activation() {
        use crate::{activation::ReLU, matrix::Matrix, activation::Activation};

        let relu = ReLU;
        let input = Matrix::from_vec(vec![vec![-2.0, 0.0, 2.0]]);
        let activated = relu.activate(&input);
        let expected = Matrix::from_vec(vec![vec![0.0, 0.0, 2.0]]);

        for (a, e) in activated.data[0].iter().zip(expected.data[0].iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn relu_derivative() {
        use crate::{activation::ReLU, matrix::Matrix, activation::Activation};

        let relu = ReLU;
        let input = Matrix::from_vec(vec![vec![-2.0, 0.0, 2.0]]);
        let derivative = relu.derivative(&input);
        let expected = Matrix::from_vec(vec![vec![0.0, 0.0, 1.0]]);

        for (d, e) in derivative.data[0].iter().zip(expected.data[0].iter()) {
            assert!((d - e).abs() < 1e-6);
        }
    }
}