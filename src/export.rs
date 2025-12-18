use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::cli::TrackConfig;
use crate::matrix::Matrix;

/// Create parent directories for a path if they don't exist
pub fn ensure_parent_dir(path: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Streams loss values to CSV during training
pub struct LossWriter {
    writer: BufWriter<File>,
    batch_count: usize,
}

impl LossWriter {
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
        ensure_parent_dir(path)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "epoch,loss")?;
        Ok(Self { writer, batch_count: 0 })
    }

    pub fn write(&mut self, epoch: usize, loss: f64) -> Result<(), Box<dyn Error>> {
        writeln!(self.writer, "{},{:.10}", epoch, loss)?;
        self.batch_count += 1;
        // Flush every 100 writes for balance between performance and durability
        if self.batch_count % 100 == 0 {
            self.writer.flush()?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), Box<dyn Error>> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Streams prediction evolution to CSV during training
pub struct PredictionWriter {
    writer: BufWriter<File>,
    batch_count: usize,
}

impl PredictionWriter {
    pub fn new(path: &str, num_samples: usize) -> Result<Self, Box<dyn Error>> {
        ensure_parent_dir(path)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        // Header: epoch,sample_0,sample_1,...
        let header = std::iter::once("epoch".to_string())
            .chain((0..num_samples).map(|i| format!("sample_{}", i)))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(writer, "{}", header)?;
        Ok(Self { writer, batch_count: 0 })
    }

    pub fn write(&mut self, epoch: usize, predictions: &[f64]) -> Result<(), Box<dyn Error>> {
        let values = std::iter::once(epoch.to_string())
            .chain(predictions.iter().map(|p| format!("{:.10}", p)))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(self.writer, "{}", values)?;
        self.batch_count += 1;
        if self.batch_count % 100 == 0 {
            self.writer.flush()?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), Box<dyn Error>> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Export decision boundary grid to CSV
pub fn export_boundary(
    path: &str,
    inputs: &Matrix,
    predictions: &Matrix,
) -> Result<(), Box<dyn Error>> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "x,y,prediction")?;
    for i in 0..inputs.rows {
        writeln!(
            writer,
            "{:.10},{:.10},{:.10}",
            inputs.data[i][0],
            inputs.data[i][1],
            predictions.data[i][0]
        )?;
    }
    writer.flush()?;
    Ok(())
}

// ============================================================================
// Training Output Directory Management
// ============================================================================

/// Training run metadata
#[derive(Serialize, Deserialize)]
pub struct Metadata {
    pub version: u32,
    pub seed: u64,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub data_file: String,
    pub architecture: Architecture,
    pub hyperparameters: Hyperparameters,
    pub tracking: TrackingConfig,
    pub results: Option<Results>,
}

#[derive(Serialize, Deserialize)]
pub struct Architecture {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: String,
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub epochs: usize,
    pub sample_rate: usize,
}

#[derive(Serialize, Deserialize)]
pub struct TrackingConfig {
    pub loss: bool,
    pub predictions: bool,
    pub boundary: bool,
    pub checkpoint: bool,
    pub boundary_resolution: usize,
    pub checkpoint_rate: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Results {
    pub final_loss: f64,
    pub total_epochs: usize,
}

/// Manages all training output to a directory
pub struct TrainingOutput {
    dir: String,
    metadata: Metadata,
    loss_writer: Option<LossWriter>,
    pred_writer: Option<PredictionWriter>,
    boundary_dir: Option<String>,
    checkpoint_dir: Option<String>,
    boundary_resolution: usize,
}

impl TrainingOutput {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dir: &str,
        seed: u64,
        data_file: &str,
        input_size: usize,
        output_size: usize,
        layers: &str,
        learning_rate: f64,
        epochs: usize,
        sample_rate: usize,
        track: &TrackConfig,
        boundary_resolution: usize,
        checkpoint_rate: usize,
        num_samples: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // Create directory structure
        fs::create_dir_all(dir)?;

        let loss_writer = if track.loss {
            Some(LossWriter::new(&format!("{}/loss.csv", dir))?)
        } else {
            None
        };

        let pred_writer = if track.predictions {
            Some(PredictionWriter::new(&format!("{}/predictions.csv", dir), num_samples)?)
        } else {
            None
        };

        let boundary_dir = if track.boundary {
            let path = format!("{}/boundary", dir);
            fs::create_dir_all(&path)?;
            Some(path)
        } else {
            None
        };

        let checkpoint_dir = if track.checkpoint {
            let path = format!("{}/checkpoints", dir);
            fs::create_dir_all(&path)?;
            Some(path)
        } else {
            None
        };

        let metadata = Metadata {
            version: 1,
            seed,
            created_at: Utc::now(),
            completed_at: None,
            data_file: data_file.to_string(),
            architecture: Architecture {
                input_size,
                output_size,
                layers: layers.to_string(),
            },
            hyperparameters: Hyperparameters {
                learning_rate,
                epochs,
                sample_rate,
            },
            tracking: TrackingConfig {
                loss: track.loss,
                predictions: track.predictions,
                boundary: track.boundary,
                checkpoint: track.checkpoint,
                boundary_resolution,
                checkpoint_rate,
            },
            results: None,
        };

        Ok(Self {
            dir: dir.to_string(),
            metadata,
            loss_writer,
            pred_writer,
            boundary_dir,
            checkpoint_dir,
            boundary_resolution,
        })
    }

    pub fn write_epoch(
        &mut self,
        epoch: usize,
        loss: f64,
        predictions: &[f64],
    ) -> Result<(), Box<dyn Error>> {
        if let Some(ref mut writer) = self.loss_writer {
            writer.write(epoch, loss)?;
        }
        if let Some(ref mut writer) = self.pred_writer {
            writer.write(epoch, predictions)?;
        }
        Ok(())
    }

    pub fn write_boundary(
        &self,
        epoch: usize,
        grid: &Matrix,
        predictions: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(ref dir) = self.boundary_dir {
            let path = format!("{}/epoch_{:05}.csv", dir, epoch);
            export_boundary(&path, grid, predictions)?;
        }
        Ok(())
    }

    pub fn write_checkpoint(
        &self,
        epoch: usize,
        network: &crate::network::Network,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(ref dir) = self.checkpoint_dir {
            let path = format!("{}/epoch_{:05}.json", dir, epoch);
            crate::model::save_model(
                network,
                &path,
                self.metadata.architecture.input_size,
                self.metadata.architecture.output_size,
                self.metadata.seed,
            )?;
        }
        Ok(())
    }

    pub fn boundary_resolution(&self) -> usize {
        self.boundary_resolution
    }

    pub fn tracking_boundary(&self) -> bool {
        self.boundary_dir.is_some()
    }

    pub fn tracking_checkpoint(&self) -> bool {
        self.checkpoint_dir.is_some()
    }

    pub fn checkpoint_rate(&self) -> usize {
        self.metadata.tracking.checkpoint_rate
    }

    pub fn finish(mut self, final_loss: f64, total_epochs: usize) -> Result<String, Box<dyn Error>> {
        // Finish streaming writers
        if let Some(writer) = self.loss_writer.take() {
            writer.finish()?;
        }
        if let Some(writer) = self.pred_writer.take() {
            writer.finish()?;
        }

        // Update and save metadata
        self.metadata.completed_at = Some(Utc::now());
        self.metadata.results = Some(Results { final_loss, total_epochs });

        let metadata_path = format!("{}/metadata.json", self.dir);
        let file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(file, &self.metadata)?;

        Ok(self.dir)
    }
}

// ============================================================================
// Loading functions for visualisation
// ============================================================================

/// Load metadata from a training output directory
pub fn load_metadata(dir: &str) -> Result<Metadata, Box<dyn Error>> {
    let path = format!("{}/metadata.json", dir);
    let file = File::open(&path)?;
    let metadata: Metadata = serde_json::from_reader(file)?;
    Ok(metadata)
}

/// List boundary files in epoch order
pub fn list_boundary_files(dir: &str) -> Result<Vec<(usize, String)>, Box<dyn Error>> {
    let boundary_dir = format!("{}/boundary", dir);
    let mut files: Vec<(usize, String)> = Vec::new();

    for entry in fs::read_dir(&boundary_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename.starts_with("epoch_") && filename.ends_with(".csv") {
                // Extract epoch number from "epoch_00000.csv"
                let epoch_str = &filename[6..filename.len() - 4];
                if let Ok(epoch) = epoch_str.parse::<usize>() {
                    files.push((epoch, path.to_string_lossy().to_string()));
                }
            }
        }
    }

    files.sort_by_key(|(epoch, _)| *epoch);
    Ok(files)
}

/// Load boundary data from CSV (predictions only)
pub fn load_boundary(path: &str) -> Result<Matrix, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut data: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        // Only need the prediction column (third column)
        let prediction: f64 = record[2].parse()?;
        data.push(vec![prediction]);
    }

    Ok(Matrix {
        rows: data.len(),
        cols: 1,
        data,
    })
}
