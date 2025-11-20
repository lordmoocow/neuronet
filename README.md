# neuronet - Neural Network CLI

A basic neural network implementation built from scratch in Rust, created as a learning project to understand the fundamentals of deep learning.

## Features

- **Configurable network** - Build networks with custom layer sizes and activations
- **Activation functions** - ReLU and Sigmoid
- **Model persistence** - Save and load trained models as JSON
- **Visualisations** - ASCII plots for loss curves and decision boundaries
- **Training/Test data** - Support for CSV and JSON input formats

## Prerequisites

- Rust 1.70 or higher (2024 edition)

## Installation

Clone the repository and build the project:

```bash
git clone <repository-url>
cd neuronet
cargo build --release
```

The binary will be available at `./target/release/neuronet`.

## Quick Start

### Train a simple XOR network

```bash
cargo run --release -- train \
  --data data/xor_data.csv \
  --target-columns 1 \
  --layers "4:relu,4:relu,1:sigmoid" \
  --save xor_model.json \
  --learning-rate 0.6 \
  --epochs 5000
```

### Test the trained model

```bash
cargo run --release -- test \
  --model xor_model.json \
  --data data/xor_test.csv
```

### Visualise a 2D decision boundary

```bash
cargo run --release -- visualise \
  --model xor_model.json \
  --resolution 40
```

## Usage

The `neuronet` CLI has three subcommands: `train`, `test`, and `visualise`.

### Training

Train a new neural network on your data:

```bash
neuronet train [OPTIONS]
```

**Required options:**

- `--data <FILE>` - Path to training data (CSV or JSON)
- `--layers <SPEC>` - Network architecture (e.g., "4:relu,4:relu,1:sigmoid")
- `--save <FILE>` - Where to save the trained model

**Optional parameters:**

- `--target-columns <N>` - Number of target columns at end of data file
- `--targets <FILE>` - Separate file containing targets
- `--learning-rate <RATE>` - Learning rate for gradient descent (default: 0.6)
- `--epochs <N>` - Number of training epochs (default: 5000)
- `--sample-rate <N>` - Record metrics every N epochs (default: 10)
- `--no-plots` - Disable terminal plots
- `--verbose` - Enable detailed output

**Layer specification format:**

The `--layers` argument defines your network architecture as a comma-separated list:

```text
"size1:activation1,size2:activation2,..."
```

For example, `"4:relu,4:relu,1:sigmoid"` creates:

- Input layer (size inferred from data)
- Hidden layer with 4 neurons and ReLU activation
- Hidden layer with 4 neurons and ReLU activation
- Output layer with 1 neuron and Sigmoid activation

Supported activations: `relu`, `sigmoid`

### Testing

Run predictions on new data with a trained model:

```bash
neuronet test --model <FILE> --data <FILE> [--verbose]
```

### Visualisation

For 2D input problems, visualise the decision boundary:

```bash
neuronet visualise --model <FILE> --resolution <N> [--verbose]
```

The visualisation generates a grid of points in the [0,1]×[0,1] space and displays the network's predictions as ASCII art.

## Data Formats

### CSV Format

Each row is a sample, columns are features. Optionally, the last N columns can be targets:

```csv
0.0,0.0,0.0
0.0,1.0,1.0
1.0,0.0,1.0
1.0,1.0,0.0
```

Use `--target-columns 1` to indicate the last column contains targets.

### JSON Format

```json
{
  "inputs": [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ],
  "targets": [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
  ]
}
```

## How It Works

This implementation includes:

1. **Matrix operations** - Custom matrix library with dot product, element-wise operations, and transpose
2. **Forward propagation** - Data flows through layers with weighted sums and activation functions
3. **Backpropagation** - Gradients are computed and propagated backward to update weights
4. **Gradient descent** - Weights and biases are updated to minimise the loss function
5. **He initialization** - Weights are initialised using He initialization for better convergence

The network uses Mean Squared Error (MSE) as the loss function and supports both ReLU and Sigmoid activation functions.

## Example Datasets

The `data/` directory includes several example datasets:

- **XOR problem** (`xor_data.csv`) - Classic non-linearly separable problem
- **AND/OR gates** (`and_data.csv`, `or_data.csv`) - Simple logic gates
- **Circle classification** (`circle_train.csv`, `circle_test.csv`) - 2D classification problem

## Contributing

This is a personal learning project, but suggestions and improvements are welcome! Feel free to open an issue or submit a pull request.

## Licence

This project is licensed under the MIT Licence - see the [LICENSE](LICENSE) file for details.
