use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::matrix::Matrix;

/// Create parent directories for a path if they don't exist
fn ensure_parent_dir(path: &str) -> Result<(), Box<dyn Error>> {
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
