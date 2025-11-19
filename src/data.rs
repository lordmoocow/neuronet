use crate::matrix::Matrix;
use std::error::Error;
use std::fs::File;
use std::path::Path;

/// Represents a dataset with inputs and optional targets
pub struct Dataset {
    pub inputs: Matrix,
    pub targets: Option<Matrix>,
}

/// Load a dataset from a file (CSV or JSON format)
///
/// For CSV files:
/// - If `target_columns` is Some, those columns are extracted as targets
/// - If `target_columns` is None, all columns are treated as inputs
/// - First row is treated as header and skipped
///
/// For JSON files:
/// - Expected format: {"inputs": [[...], ...], "targets": [[...], ...]}
/// - Or just: [[...], ...] for inputs only
pub fn load_data(
    path: &str,
    target_columns: Option<usize>,
) -> Result<Dataset, Box<dyn Error>> {
    let path_obj = Path::new(path);
    let extension = path_obj
        .extension()
        .and_then(|s| s.to_str())
        .ok_or("File must have an extension (.csv or .json)")?;

    match extension.to_lowercase().as_str() {
        "csv" => load_csv(path, target_columns),
        "json" => load_json(path),
        _ => Err(format!("Unsupported file format: {}", extension).into()),
    }
}

/// Load data from a CSV file
fn load_csv(path: &str, target_columns: Option<usize>) -> Result<Dataset, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(file);

    let mut all_data: Vec<Vec<f64>> = Vec::new();

    // Read all records
    for result in reader.records() {
        let record = result?;
        let row: Result<Vec<f64>, _> = record.iter().map(|field| field.parse::<f64>()).collect();
        all_data.push(row?);
    }

    if all_data.is_empty() {
        return Err("CSV file is empty".into());
    }

    // Split into inputs and targets if target_columns specified
    match target_columns {
        Some(num_targets) => {
            let num_features = all_data[0].len();
            if num_targets >= num_features {
                return Err(format!(
                    "Target columns ({}) must be less than total columns ({})",
                    num_targets, num_features
                )
                .into());
            }

            let split_idx = num_features - num_targets;
            let mut inputs_data = Vec::new();
            let mut targets_data = Vec::new();

            for row in all_data {
                inputs_data.push(row[..split_idx].to_vec());
                targets_data.push(row[split_idx..].to_vec());
            }

            Ok(Dataset {
                inputs: Matrix::from_vec(inputs_data),
                targets: Some(Matrix::from_vec(targets_data)),
            })
        }
        None => Ok(Dataset {
            inputs: Matrix::from_vec(all_data),
            targets: None,
        }),
    }
}

/// Load data from a JSON file
fn load_json(path: &str) -> Result<Dataset, Box<dyn Error>> {
    let file = File::open(path)?;
    let json_value: serde_json::Value = serde_json::from_reader(file)?;

    // Try to parse as structured format with inputs and targets
    if let Some(obj) = json_value.as_object() {
        if let Some(inputs) = obj.get("inputs") {
            let inputs_data: Vec<Vec<f64>> = serde_json::from_value(inputs.clone())?;
            let targets_data = if let Some(targets) = obj.get("targets") {
                Some(Matrix::from_vec(serde_json::from_value(targets.clone())?))
            } else {
                None
            };

            return Ok(Dataset {
                inputs: Matrix::from_vec(inputs_data),
                targets: targets_data,
            });
        }
    }

    // Try to parse as simple array (inputs only)
    if let Ok(inputs_data) = serde_json::from_value::<Vec<Vec<f64>>>(json_value) {
        return Ok(Dataset {
            inputs: Matrix::from_vec(inputs_data),
            targets: None,
        });
    }

    Err("JSON format not recognized. Expected {\"inputs\": [...], \"targets\": [...]} or [[...], ...]".into())
}

/// Load targets from a separate file
pub fn load_targets(path: &str) -> Result<Matrix, Box<dyn Error>> {
    let dataset = load_data(path, None)?;
    Ok(dataset.inputs) // Targets file is just treated as inputs
}
