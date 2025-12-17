use crate::{layer::Layer, loss::Loss, matrix::Matrix};

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Get reference to layers (for serialization)
    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    fn forward(&mut self, input: &Matrix) -> Matrix {
        // Forward pass through each layer in sequence
        // starting with the input matrix
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            // Each layer processes the output of the previous layer
            output = layer.forward(&output);
        }
        output
    }

    fn backward(&mut self, loss_grad: &Matrix, learning_rate: f64) {
        // Backpropagate the loss gradient through each layer in reverse order
        // starting with the gradient from the loss function
        let mut grad = loss_grad.clone();
        for layer in self.layers.iter_mut().rev() {
            // Each layer updates its weights and biases based on the incoming gradient
            // and produces a gradient for the previous layer
            grad = layer.backward(&grad, learning_rate);
        }
    }

    pub fn predict(&mut self, input: &Matrix) -> Matrix {
        // Prediction is just a forward pass with current weights without any training
        self.forward(input)
    }

    pub fn train_batch(&mut self, inputs: &Matrix, targets: &Matrix, learning_rate: f64, loss_fn: &dyn Loss) -> (f64, Matrix) {
        // Forward pass dertermines our expected output for given inputs
        // with respect to current network weights and biases
        let predictions = self.forward(inputs);

        // From our prediction we can compute the loss with respect to the targets
        // and also compute the gradient of that loss to use in backpropagation
        let loss = loss_fn.loss(&predictions, targets);
        let loss_grad = loss_fn.derivative(&predictions, targets);

        // Backward pass uses the loss gradient to update weights and biases
        // with the aim of reducing future loss
        self.backward(&loss_grad, learning_rate);

        // Return current loss value for monitoring
        (loss, predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn new_add_layer() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut network = Network::new();
        assert_eq!(network.layers.len(), 0);

        let layer1 = Layer::new(2, 3, Box::new(crate::activation::Sigmoid), &mut rng);
        network.add_layer(layer1);
        assert_eq!(network.layers.len(), 1);

        let layer2 = Layer::new(3, 1, Box::new(crate::activation::Sigmoid), &mut rng);
        network.add_layer(layer2);
        assert_eq!(network.layers.len(), 2);
    }

    #[test]
    fn forward_predict_shapes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut network = Network::new();
        network.add_layer(Layer::new(2, 10, Box::new(crate::activation::Sigmoid), &mut rng));
        network.add_layer(Layer::new(10, 1, Box::new(crate::activation::Sigmoid), &mut rng));

        let input = Matrix::from_vec(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ]);

        let output_forward = network.forward(&input);
        assert_eq!(output_forward.rows, 2);
        assert_eq!(output_forward.cols, 1);

        let output_predict = network.predict(&input);
        assert_eq!(output_predict.rows, 2);
        assert_eq!(output_predict.cols, 1);
    }

    #[test]
    fn train_batch_reduces_loss() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut network = Network::new();
        network.add_layer(Layer::new(2, 3, Box::new(crate::activation::Sigmoid), &mut rng));
        network.add_layer(Layer::new(3, 1, Box::new(crate::activation::Sigmoid), &mut rng));

        let inputs = Matrix::from_vec(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ]);
        let targets = Matrix::from_vec(vec![
            vec![0.0],
            vec![1.0],
        ]);

        let loss_fn = crate::loss::MSE;

        let (initial_loss, _) = network.train_batch(&inputs, &targets, 0.5, &loss_fn);
        for _ in 0..10 {
            network.train_batch(&inputs, &targets, 0.5, &loss_fn);
        }
        let (final_loss, _) = network.train_batch(&inputs, &targets, 0.5, &loss_fn);

        assert!(final_loss < initial_loss);
    }
}