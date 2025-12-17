use clap::{Parser, Subcommand};

/// Neural Network CLI Tool - Train and test neural networks from scratch
#[derive(Parser, Debug)]
#[command(name = "neuronet")]
#[command(about = "A simple neural network implementation with configurable architecture", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a new neural network model
    Train(TrainArgs),
    /// Test a trained model on new data
    Test(TestArgs),
    /// Visualise a trained model on new data (2D)
    Visualise(VisualiseArgs),
}

/// Training arguments
#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Path to training data file (CSV or JSON format)
    ///
    /// CSV format: Each row is a sample, columns are features (and optionally targets at the end)
    /// JSON format: {"inputs": [[...], ...], "targets": [[...], ...]}
    #[arg(short, long, value_name = "FILE")]
    pub data: String,

    /// Path to separate targets file (CSV or JSON)
    ///
    /// If not specified, use --target-columns to extract targets from data file
    #[arg(short, long, value_name = "FILE")]
    pub targets: Option<String>,

    /// Number of target columns at the end of the data file
    ///
    /// If specified, the last N columns of the data file will be used as targets
    /// Ignored if --targets is provided
    #[arg(long, value_name = "N")]
    pub target_columns: Option<usize>,

    /// Network layer specification: "size1:activation1,size2:activation2,..."
    ///
    /// Example: "4:relu,4:relu,1:sigmoid" creates 3 layers
    /// Activations: relu, sigmoid
    /// First layer input size is inferred from data
    #[arg(short, long, value_name = "SPEC")]
    pub layers: String,

    /// Path to save the trained model (JSON format)
    #[arg(short, long, value_name = "FILE")]
    pub save: String,

    /// Learning rate for gradient descent
    #[arg(short = 'r', long, default_value = "0.6", value_name = "RATE")]
    pub learning_rate: f64,

    /// Number of training epochs
    #[arg(short, long, default_value = "5000", value_name = "N")]
    pub epochs: usize,

    /// Random seed for weight initialization (auto-generated if not provided)
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,

    /// Sample rate for tracking loss and predictions (record every Nth epoch)
    #[arg(long, default_value = "10", value_name = "N")]
    pub sample_rate: usize,

    /// Disable terminal plots
    #[arg(long)]
    pub no_plots: bool,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Testing arguments
#[derive(Parser, Debug)]
pub struct TestArgs {
    /// Path to the trained model file (JSON format)
    #[arg(short, long, value_name = "FILE")]
    pub model: String,

    /// Path to test data file (CSV or JSON format)
    ///
    /// CSV format: Each row is a sample, columns are features
    /// JSON format: [[...], ...] or {"inputs": [[...], ...]}
    #[arg(short, long, value_name = "FILE")]
    pub data: String,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Parser, Debug)]
pub struct VisualiseArgs {
    /// Path to the trained model file (JSON format)
    #[arg(short, long, value_name="FILE")]
    pub model: String,

    #[arg(short, long)]
    pub resolution: usize,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Represents a single layer specification
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub size: usize,
    pub activation: String,
}

/// Parse the layer specification string into a vector of LayerSpec
///
/// Format: "size1:activation1,size2:activation2,..."
/// Example: "4:relu,4:relu,1:sigmoid"
pub fn parse_layers(spec: &str) -> Result<Vec<LayerSpec>, String> {
    let mut layers = Vec::new();

    for layer_str in spec.split(',') {
        let parts: Vec<&str> = layer_str.trim().split(':').collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid layer specification '{}'. Expected format: 'size:activation'",
                layer_str
            ));
        }

        let size = parts[0]
            .parse::<usize>()
            .map_err(|_| format!("Invalid layer size '{}'. Must be a positive integer", parts[0]))?;

        if size == 0 {
            return Err(format!("Layer size must be greater than 0, got {}", size));
        }

        let activation = parts[1].trim().to_lowercase();
        if activation != "relu" && activation != "sigmoid" {
            return Err(format!(
                "Unknown activation function '{}'. Supported: relu, sigmoid",
                parts[1]
            ));
        }

        layers.push(LayerSpec { size, activation });
    }

    if layers.is_empty() {
        return Err("Layer specification cannot be empty".to_string());
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_layers() {
        let result = parse_layers("4:relu,4:relu,1:sigmoid").unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].size, 4);
        assert_eq!(result[0].activation, "relu");
        assert_eq!(result[2].size, 1);
        assert_eq!(result[2].activation, "sigmoid");
    }

    #[test]
    fn test_parse_layers_invalid() {
        assert!(parse_layers("4:relu,invalid").is_err());
        assert!(parse_layers("0:relu").is_err());
        assert!(parse_layers("4:unknown").is_err());
        assert!(parse_layers("").is_err());
    }
}
