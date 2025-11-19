use std::fmt::Debug;
use rand::Rng;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.data {
            writeln!(f, "{:?}", row)?;
        }
        Ok(())
    }
}

impl Matrix {
    /// Creates a new matrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// Creates a matrix from a 2D vector.
    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        // data is already in the correct format so we just need to get the dimensions
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        // He initialisation scales the values based on the number of inputs
        let scale = (2.0 / rows as f64).sqrt();

        let mut rng = rand::rng();
        let mut data = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                data[i][j] = rng.random_range(-1.0..1.0) * scale;
            }
        }
        Self { rows, cols, data }
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, 
          "Matrix dimensions incompatible: ({}, {}) × ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, other.cols);

        // for each row in self
        for i in 0..self.rows {
            // for each column in other
            for j in 0..other.cols {
                // sum the product
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }

        result
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, 
          "Matrix dimensions incompatible for element-wise multiplication: ({}, {}) and ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );
        assert_eq!(self.cols, other.cols, 
          "Matrix dimensions incompatible for element-wise multiplication: ({}, {}) and ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }

        result
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, 
          "Matrix dimensions incompatible for addition: ({}, {}) and ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );
        assert_eq!(self.cols, other.cols, 
          "Matrix dimensions incompatible for addition: ({}, {}) and ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        result
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, 
          "Matrix dimensions incompatible for subtraction: ({}, {}) and ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );
        assert_eq!(self.cols, other.cols, 
          "Matrix dimensions incompatible for subtraction: ({}, {}) and ({}, {})",
          self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        result
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }

        result
    }

    pub fn map<F>(&self, f: F) -> Matrix where F: Fn(f64) -> f64 {
        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = f(self.data[i][j]);
            }
        }

        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn from_vec() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let matrix = Matrix::from_vec(data.clone());
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn zeros() {
        let matrix = Matrix::zeros(2, 3);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]);
    }

    #[test]
    fn dot_product() {
        let a = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        let b = Matrix::from_vec(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);

        let c = a.dot(&b);
        assert_eq!(c.data, vec![
            vec![19.0, 22.0],
            vec![43.0, 50.0],
        ]);
    }

    #[test]
    fn addition() {
        let a = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        let b = Matrix::from_vec(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);

        let c = a.add(&b);
        assert_eq!(c.data, vec![
            vec![6.0, 8.0],
            vec![10.0, 12.0],
        ]);
    }

    #[test]
    fn subtraction() {
        let a = Matrix::from_vec(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);

        let b = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        let c = a.subtract(&b);
        assert_eq!(c.data, vec![
            vec![4.0, 4.0],
            vec![4.0, 4.0],
        ]);
    }

    #[test]
    fn multiplication() {
        let a = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        let b = Matrix::from_vec(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);

        let c = a.multiply(&b);
        assert_eq!(c.data, vec![
            vec![5.0, 12.0],
            vec![21.0, 32.0],
        ]);
    }

    #[test]
    fn transpose() {
        let a = Matrix::from_vec(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);

        let b = a.transpose();
        assert_eq!(b.data, vec![
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
        ]);
    }

    #[test]
    fn scale() {
        let a = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        let b = a.scale(2.0);
        assert_eq!(b.data, vec![
            vec![2.0, 4.0],
            vec![6.0, 8.0],
        ]);
    }

    #[test]
    fn map() {
        let a = Matrix::from_vec(vec![
            vec![1.0, -2.0],
            vec![-3.0, 4.0],
        ]);

        let b = a.map(|x| x.max(0.0)); 
        assert_eq!(b.data, vec![
            vec![1.0, 0.0],
            vec![0.0, 4.0],
        ]);
    }
}