use crate::matrix::Matrix;
use crate::activation::Activation;

pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: Box<dyn Activation>,

    // Cache for backpropagation
    last_input: Option<Matrix>,
    last_activation: Option<Matrix>,
}

impl Layer {
    /// Create a new layer
    /// input_size: number of inputs to this layer
    /// output_size: number of neurons in this layer
    pub fn new(input_size: usize, output_size: usize, activation: Box<dyn Activation>) -> Self {
        Self {
            // weights matrix represents a row per input/feature and a column per neuron
            weights: Matrix::random(input_size, output_size),
            // biases is a single row which contains a value for each neuron
            biases: Matrix::zeros(1, output_size),
            // the activation function for this layer
            activation,

            // cache data for backpropagation
            last_input: None,
            last_activation: None,
        }
    }

    /// Forward pass through the layer
    /// input shape: (batch_size, input_size)
    /// Returns: activated output, shape (batch_size, output_size)
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        // combine inputs and weights
        // this provides every neuron with a weighted sum of the inputs
        let z = input.dot(&self.weights);

        // add bias to each neuron
        let z = self.add_bias(&z);

        // apply activation function
        let activated = self.activation.activate(&z);

        // Cache values for backprop
        self.last_input = Some(input.clone());
        self.last_activation = Some(activated.clone());

        activated
    }

    pub fn backward(&mut self, grad_output: &Matrix, learning_rate: f64) -> Matrix {
        let activation = self.last_activation.as_ref().expect("last_activation missing. Forward pass not called before backward?");
        let input = self.last_input.as_ref().expect("last_input missing. Forward pass not called before backward?");

        // First we use the cached activation from forward pass to compute the deriviative
        let derivative = self.activation.derivative(activation);
        // We can then use this to compute the gradient through the activation function
        let grad_activation = grad_output.multiply(&derivative);

        // Using the output of the next layer, going in reverse we need to transpose the matrix
        // as the output shape is now our input.

        // compute the gradient of the weights
        let grad_weights = input.transpose().dot(&grad_activation);
        // compute the gradient of the biases 
        // (this is effectively a vector stored as a matrix for simplicity, we just sum the gradients into single row)
        let grad_biases = self.sum_rows(&grad_activation);
        // compute the gradient to pass to the previous layer
        // This time we inverse the weights shape in order to reshape our gradient to the output of the previous layer
        let grad_input = grad_activation.dot(&self.weights.transpose());

        // We subtract the gradients scaled by the learning rate from the weights and biases
        // the learning rate controls how big of a step we take in the direction of the gradient
        // this is the "gradient descent"
        self.weights = self.weights.subtract(&grad_weights.scale(learning_rate));
        self.biases = self.biases.subtract(&grad_biases.scale(learning_rate));

        grad_input
    }

    /// Helper method: Add bias to each row of the matrix
    /// z: matrix of shape (batch_size, output_size)
    /// Returns: z with bias added to each row
    fn add_bias(&self, z: &Matrix) -> Matrix {
        let mut result = z.clone();
        for i in 0..result.rows {
            for j in 0..result.cols {
                // bias is a single row, so always index row 0
                // not sure if bias needs to be a matrix?
                result.data[i][j] += self.biases.data[0][j];
            }
        }
        result
    }

    /// Helper method: Sum the rows of a matrix into a single row matrix
    /// matrix: input matrix of shape (batch_size, output_size)
    /// Returns: matrix of shape (1, output_size) containing the sum of each column
    fn sum_rows(&self, matrix: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(1, matrix.cols);
        for j in 0..matrix.cols {
            let mut sum = 0.0;
            for i in 0..matrix.rows {
                sum += matrix.data[i][j];
            }
            result.data[0][j] = sum;
        }
        result
    }

    /// Get the weights matrix (for serialization)
    pub fn get_weights(&self) -> &Matrix {
        &self.weights
    }

    /// Get the biases matrix (for serialization)
    pub fn get_biases(&self) -> &Matrix {
        &self.biases
    }

    /// Get the activation function name (for serialization)
    pub fn get_activation_name(&self) -> &str {
        self.activation.name()
    }

    /// Set the weights matrix (for deserialization)
    pub fn set_weights(&mut self, weights: Matrix) {
        self.weights = weights;
    }

    /// Set the biases matrix (for deserialization)
    pub fn set_biases(&mut self, biases: Matrix) {
        self.biases = biases;
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn forward() {
        use crate::{layer::Layer, activation::Sigmoid, matrix::Matrix};

        let mut layer = Layer::new(2, 2, Box::new(Sigmoid));

        // Manually set weights and biases for predictable output
        layer.weights = Matrix::from_vec(vec![
            vec![0.5, -0.5],
            vec![0.3, 0.8],
        ]);
        layer.biases = Matrix::from_vec(vec![
            vec![0.1, -0.1],
        ]);

        let input = Matrix::from_vec(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]);

        let output = layer.forward(&input);

        let expected = Matrix::from_vec(vec![
            vec![0.6456563062257954, 0.35434369377420455],
            vec![0.598687660112452, 0.6681877721681662],
        ]);
        
        println!("Output:\n{:?}\n", output);
        println!("Expected:\n{:?}\n", expected);

        for i in 0..output.rows {
            for j in 0..output.cols {
                assert!(
                    (output.data[i][j] - expected.data[i][j]).abs() < f64::EPSILON, 
                    "Mismatch at ({}, {}): got {}, expected {}", i, j, output.data[i][j], expected.data[i][j]
                );
            }
        }
    }

    #[test]
    fn backward() {
        use crate::{layer::Layer, activation::Sigmoid, matrix::Matrix};

        let mut layer = Layer::new(2, 2, Box::new(Sigmoid));
        let starting_weights = Matrix::from_vec(vec![
            vec![0.5, -0.5],
            vec![0.3, 0.8],
        ]);

        // Manually set weights and biases for predictable output
        layer.weights = starting_weights.clone();
        layer.biases = Matrix::from_vec(vec![
            vec![0.1, -0.1],
        ]);

        let input = Matrix::from_vec(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]);

        // Forward pass
        let _ = layer.forward(&input);

        // Fake gradient from next layer
        let grad_output = Matrix::from_vec(vec![
            vec![0.1, -0.2],
            vec![-0.1, 0.15],
        ]);

        let learning_rate = 0.1;
        let grad_input = layer.backward(&grad_output, learning_rate);

        // Check shapes
        assert_eq!(grad_input.rows, input.rows);
        assert_eq!(grad_input.cols, input.cols);

        // Assert weights changed
        assert_ne!(starting_weights.data, layer.weights.data);
    }
}