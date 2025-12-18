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

    /// Path to save the trained model (ignored if --output-dir is set)
    #[arg(short, long, value_name = "FILE")]
    pub save: Option<String>,

    /// Output directory for training artifacts (model, metrics, checkpoints)
    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<String>,

    /// What to track: comma-separated list of loss,predictions,boundary,checkpoint
    #[arg(long, value_name = "LIST", default_value = "loss,predictions")]
    pub track: String,

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

    /// Resolution for boundary grid (points per axis)
    #[arg(long, value_name = "N", default_value = "100")]
    pub boundary_resolution: usize,

    /// Save model checkpoint every N epochs (defaults to sample_rate * 100)
    #[arg(long, value_name = "N")]
    pub checkpoint_rate: Option<usize>,

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
    /// Path to trained model file (for static visualisation)
    #[arg(short, long, value_name = "FILE", conflicts_with = "dir")]
    pub model: Option<String>,

    /// Path to training output directory (for animated visualisation)
    #[arg(long, value_name = "DIR", conflicts_with = "model")]
    pub dir: Option<String>,

    /// Resolution for static visualisation (ignored with --dir)
    #[arg(short, long, default_value = "40")]
    pub resolution: usize,

    /// Delay between frames in milliseconds (animation mode)
    #[arg(long, default_value = "100")]
    pub delay: u64,

    /// Loop animation continuously
    #[arg(long)]
    pub loop_animation: bool,

    /// Start animation from specific epoch
    #[arg(long)]
    pub start_epoch: Option<usize>,

    /// End animation at specific epoch
    #[arg(long)]
    pub end_epoch: Option<usize>,

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

/// Configuration for what to track during training
#[derive(Debug, Default, Clone)]
pub struct TrackConfig {
    pub loss: bool,
    pub predictions: bool,
    pub boundary: bool,
    pub checkpoint: bool,
}

/// Parse track specification into flags
///
/// Format: "loss,predictions,boundary,checkpoint" (comma-separated)
pub fn parse_track(spec: &str) -> Result<TrackConfig, String> {
    let mut config = TrackConfig::default();
    for item in spec.split(',') {
        match item.trim().to_lowercase().as_str() {
            "loss" => config.loss = true,
            "predictions" => config.predictions = true,
            "boundary" => config.boundary = true,
            "checkpoint" => config.checkpoint = true,
            "" => {} // ignore empty
            other => return Err(format!(
                "Unknown track option: '{}'. Valid: loss,predictions,boundary,checkpoint",
                other
            )),
        }
    }
    Ok(config)
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
