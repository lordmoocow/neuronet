use std::{error::Error, fs::File, io::Write};

use serde::{Deserialize, Serialize};

use crate::activation::{ReLU, Sigmoid};
use crate::export::ensure_parent_dir;
use crate::layer::Layer;
use crate::matrix::Matrix;
use crate::network::Network;

/// Serializable representation of a neural network layer
#[derive(Serialize, Deserialize)]
pub struct SerializableLayer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: String,
}

/// Serializable representation of a neural network model
#[derive(Serialize, Deserialize)]
pub struct SerializableModel {
    pub layers: Vec<SerializableLayer>,
    pub metadata: ModelMetadata,
}

/// Metadata about the trained model
#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    pub created_at: String,
    pub input_size: usize,
    pub output_size: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl SerializableModel {
    /// Create a serializable model from a trained network
    pub fn from_network(network: &Network, input_size: usize, output_size: usize, seed: u64) -> Self {
        let mut serializable_layers = Vec::new();

        for layer in network.get_layers() {
            serializable_layers.push(SerializableLayer {
                weights: layer.get_weights().clone(),
                biases: layer.get_biases().clone(),
                activation: layer.get_activation_name().to_string(),
            });
        }

        Self {
            layers: serializable_layers,
            metadata: ModelMetadata {
                created_at: get_current_timestamp(),
                input_size,
                output_size,
                num_layers: network.get_layers().len(),
                seed: Some(seed),
            },
        }
    }

    /// Convert a serializable model back to a network
    pub fn to_network(&self) -> Result<Network, Box<dyn Error>> {
        let mut network = Network::new();

        for layer_data in &self.layers {
            let activation: Box<dyn crate::activation::Activation> = match layer_data.activation.as_str() {
                "sigmoid" => Box::new(Sigmoid),
                "relu" => Box::new(ReLU),
                _ => return Err(format!("Unknown activation function: {}", layer_data.activation).into()),
            };

            // Create a layer from the saved weights and biases
            let layer = Layer::from_weights(
                layer_data.weights.clone(),
                layer_data.biases.clone(),
                activation,
            );

            network.add_layer(layer);
        }

        Ok(network)
    }
}

/// Save a trained network to a JSON file
pub fn save_model(
    network: &Network,
    path: &str,
    input_size: usize,
    output_size: usize,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    ensure_parent_dir(path)?;
    let model = SerializableModel::from_network(network, input_size, output_size, seed);
    let json = serde_json::to_string_pretty(&model)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Load a trained network from a JSON file
pub fn load_model(path: &str) -> Result<Network, Box<dyn Error>> {
    let file = File::open(path)?;
    let model: SerializableModel = serde_json::from_reader(file)?;
    model.to_network()
}

fn get_current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    format!("{}", duration.as_secs())
}
