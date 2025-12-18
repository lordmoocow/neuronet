use clap::Parser;
use std::process;
use rand::SeedableRng;

use crate::{
    activation::{ReLU, Sigmoid},
    cli::{Cli, Commands, parse_layers},
    data::{load_data, load_targets},
    layer::Layer,
    loss::MSE,
    matrix::Matrix,
    model::{save_model, load_model},
    network::Network,
};

#[allow(dead_code)]

mod matrix;
mod activation;
mod layer;
mod network;
mod loss;
mod cli;
mod data;
mod model;
mod export;

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Train(train_args) => handle_train(train_args),
        Commands::Test(test_args) => handle_test(test_args),
        Commands::Visualise(vis_args) => handle_visualisation(vis_args),
    }
}

fn handle_train(args: cli::TrainArgs) {
    // Generate or use provided seed for reproducible weight initialization
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);
    println!("Neural Network Training (seed: {})\n", seed);

    // Create seeded RNG for weight initialization
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Load data
    let dataset = match load_data(&args.data, args.target_columns) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading data from '{}': {}", args.data, e);
            process::exit(1);
        }
    };

    // Load or extract targets
    let targets = if let Some(targets_path) = &args.targets {
        match load_targets(targets_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error loading targets from '{}': {}", targets_path, e);
                process::exit(1);
            }
        }
    } else if let Some(t) = dataset.targets {
        t
    } else {
        eprintln!("Error: No targets provided. Use --targets FILE or --target-columns N");
        process::exit(1);
    };

    let inputs = dataset.inputs;

    // Validate data dimensions
    if inputs.rows != targets.rows {
        eprintln!(
            "Error: Number of input samples ({}) doesn't match number of target samples ({})",
            inputs.rows, targets.rows
        );
        process::exit(1);
    }

    if args.verbose {
        println!("Loaded {} samples with {} features and {} targets",
            inputs.rows, inputs.cols, targets.cols);
    }

    // Parse layer specification
    let layer_specs = match parse_layers(&args.layers) {
        Ok(specs) => specs,
        Err(e) => {
            eprintln!("Error parsing layers: {}", e);
            process::exit(1);
        }
    };

    // Build the neural network
    let mut network = Network::new();
    let mut input_size = inputs.cols;

    for spec in &layer_specs {
        let activation: Box<dyn crate::activation::Activation> = match spec.activation.as_str() {
            "sigmoid" => Box::new(Sigmoid),
            "relu" => Box::new(ReLU),
            _ => unreachable!("Activation already validated"),
        };
        network.add_layer(Layer::new(input_size, spec.size, activation, &mut rng));
        input_size = spec.size;
    }

    // Validate output layer matches target dimensions
    if input_size != targets.cols {
        eprintln!(
            "Error: Last layer size ({}) doesn't match target dimensions ({})",
            input_size, targets.cols
        );
        process::exit(1);
    }

    if args.verbose {
        println!("Network architecture: {} -> {} layers -> {}",
            inputs.cols,
            layer_specs.len(),
            targets.cols
        );
    }

    // Training hyperparameters
    let loss_fn = MSE;

    // Initialize streaming writers if requested
    let mut loss_writer = match &args.output_loss {
        Some(path) => match export::LossWriter::new(path) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("Error creating loss output file '{}': {}", path, e);
                process::exit(1);
            }
        },
        None => None,
    };

    let mut pred_writer = match &args.output_predictions {
        Some(path) => match export::PredictionWriter::new(path, inputs.rows) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("Error creating predictions output file '{}': {}", path, e);
                process::exit(1);
            }
        },
        None => None,
    };

    // Training loop - streaming to files, no in-memory accumulation
    eprint!("Training...0%");

    for epoch in 0..args.epochs {
        // Execute training step
        let (loss, prediction) = network.train_batch(&inputs, &targets, args.learning_rate, &loss_fn);

        // Progress indicator
        if epoch % (args.epochs / 33.max(1)) == 0 && epoch != 0 {
            eprint!(".");
        }
        if epoch % (args.epochs / 10.max(1)) == 0 && epoch != 0 {
            eprint!("{}%", (epoch * 100) / args.epochs);
        }

        // Stream to files at sample rate
        if epoch % args.sample_rate == 0 || epoch == args.epochs - 1 {
            if let Some(ref mut writer) = loss_writer {
                if let Err(e) = writer.write(epoch, loss) {
                    eprintln!("Error writing loss data: {}", e);
                    process::exit(1);
                }
            }
            if let Some(ref mut writer) = pred_writer {
                // Extract first output of each sample's prediction
                let preds: Vec<f64> = prediction.data.iter().map(|row| row[0]).collect();
                if let Err(e) = writer.write(epoch, &preds) {
                    eprintln!("Error writing prediction data: {}", e);
                    process::exit(1);
                }
            }
        }
    }

    eprintln!("100%!");

    // Finish streaming writers
    if let Some(writer) = loss_writer {
        if let Err(e) = writer.finish() {
            eprintln!("Error finalizing loss output: {}", e);
            process::exit(1);
        }
        println!("Loss data written to '{}'", args.output_loss.as_ref().unwrap());
    }
    if let Some(writer) = pred_writer {
        if let Err(e) = writer.finish() {
            eprintln!("Error finalizing predictions output: {}", e);
            process::exit(1);
        }
        println!("Prediction data written to '{}'", args.output_predictions.as_ref().unwrap());
    }

    // After training, test predictions
    println!("\nFinal Predictions:");
    let final_predictions = network.predict(&inputs);

    // Print each input/target/prediction
    for i in 0..inputs.rows {
        let input = &inputs.data[i];
        let target_vec = &targets.data[i];
        let prediction_vec = &final_predictions.data[i];

        if args.verbose {
            println!("Input: {:?}", input);
            println!("  Target:     {:?}", target_vec);
            println!("  Prediction: {:?}", prediction_vec);
        } else {
            // Compact format for single-output problems
            if target_vec.len() == 1 {
                println!("Input: {:?} → Target: {:.1}, Prediction: {:.4}",
                    input, target_vec[0], prediction_vec[0]);
            } else {
                println!("Input: {:?}", input);
                println!("  Target:     {:?}", target_vec);
                println!("  Prediction: {:?}", prediction_vec);
            }
        }
    }

    // Generate decision boundary if requested (only for 2D input networks)
    if let Some(output_path) = &args.output_boundary {
        if inputs.cols == 2 {
            let grid = generate_2d_grid(args.output_boundary_resolution);
            let boundary_predictions = network.predict(&grid);
            if let Err(e) = export::export_boundary(output_path, &grid, &boundary_predictions) {
                eprintln!("Error exporting boundary: {}", e);
                process::exit(1);
            }
            println!("Decision boundary exported to '{}'", output_path);
        } else {
            eprintln!("Warning: --output-boundary requires 2D input data, skipping");
        }
    }

    // Save the trained model
    println!("\nSaving model to '{}'...", args.save);
    if let Err(e) = save_model(&network, &args.save, inputs.cols, targets.cols, seed) {
        eprintln!("Error saving model: {}", e);
        process::exit(1);
    }

    println!("Model saved successfully!");
    if args.verbose {
        println!("Training complete!");
    }
}

fn handle_test(args: cli::TestArgs) {
    if args.verbose {
        println!("Neural Network Testing\n");
    }

    // Load the trained model
    if args.verbose {
        println!("Loading model from '{}'...", args.model);
    }

    let mut network = match load_model(&args.model) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Error loading model from '{}': {}", args.model, e);
            process::exit(1);
        }
    };

    if args.verbose {
        println!("Model loaded successfully!");
    }

    // Load test data (no targets required)
    let dataset = match load_data(&args.data, None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading test data from '{}': {}", args.data, e);
            process::exit(1);
        }
    };

    let inputs = dataset.inputs;

    if args.verbose {
        println!("Loaded {} test samples with {} features\n", inputs.rows, inputs.cols);
    }

    // Run predictions
    let predictions = network.predict(&inputs);

    // Output predictions only
    println!("Predictions:");
    for i in 0..inputs.rows {
        let input = &inputs.data[i];
        let prediction_vec = &predictions.data[i];

        if args.verbose {
            println!("Input: {:?}", input);
            println!("  Prediction: {:?}", prediction_vec);
        } else {
            // Compact format
            if prediction_vec.len() == 1 {
                println!("{:.4}", prediction_vec[0]);
            } else {
                println!("{:?}", prediction_vec);
            }
        }
    }
}

fn handle_visualisation(args: cli::VisualiseArgs) {
    if args.verbose {
        println!("Neural Network 2D Visualisation\n");
    }

    // Load the trained model
    if args.verbose {
        println!("Loading model from '{}'...", args.model);
    }

    let mut network = match load_model(&args.model) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Error loading model from '{}': {}", args.model, e);
            process::exit(1);
        }
    };

    if args.verbose {
        println!("Model loaded successfully!");
    }

    // Generate 2D data to visualise
    let inputs = generate_2d_grid(args.resolution);

    if args.verbose {
        println!("Generated {} test samples with {} features\n", inputs.rows, inputs.cols);
    }

    // Run predictions
    let predictions = network.predict(&inputs);

    // Display the decision boundary
    display_decision_boundary(&predictions, args.resolution, args.verbose);
}

fn generate_2d_grid(resolution: usize) -> Matrix {
    let step = 1.0 / (resolution - 1) as f64;
    let mut grid = Matrix::zeros(resolution * resolution, 2);
    for i in 0..resolution {
        let x = step * i as f64;
        for j in 0..resolution {
            grid.data[i * resolution + j][0] = x;
            grid.data[i * resolution + j][1] = step * j as f64;
        }
    }
    grid
}

fn display_decision_boundary(predictions: &Matrix, resolution: usize, verbose: bool) {
    println!("\nDecision Boundary Visualization");
    println!("(Input space from 0.0 to 1.0 in both dimensions)\n");

    if verbose {
        println!("Resolution: {}x{} = {} points", resolution, resolution, predictions.rows);
        println!();
    }

    // Display the grid (reversed so y increases upward like a normal graph)
    // Color: Red (0.0) → Black (0.5 uncertain) → Blue (1.0)
    // Density: confident = solid, uncertain = sparse
    for i in (0..resolution).rev() {
        for j in 0..resolution {
            let index = i * resolution + j;
            let value = predictions.data[index][0];

            // Confidence: 0.0 at boundary (0.5), 1.0 at extremes (0.0 or 1.0)
            let confidence = (value - 0.5).abs() * 2.0;

            // Red (0.0) → Black (0.5) → Blue (1.0) gradient
            let (r, g, b) = if value < 0.5 {
                // Black to Red: red increases with distance from 0.5
                let t = (0.5 - value) * 2.0; // 0.0 to 1.0
                ((255.0 * t) as u8, 0, 0)
            } else {
                // Black to Blue: blue increases with distance from 0.5
                let t = (value - 0.5) * 2.0; // 0.0 to 1.0
                (0, 0, (255.0 * t) as u8)
            };

            // Symbol based on confidence (doubled for better aspect ratio)
            let symbol = match confidence {
                c if c > 0.8 => "██",
                c if c > 0.6 => "▓▓",
                c if c > 0.4 => "▒▒",
                c if c > 0.2 => "░░",
                _ => "  ",
            };

            // ANSI: foreground color with symbol
            print!("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, symbol);
        }
        println!();
    }

    println!("\nLegend:");
    println!("  \x1b[38;2;255;0;0m██\x1b[0m = 0.0 (strongly class 0)");
    println!("  \x1b[38;2;80;0;0m░░\x1b[0m / \x1b[38;2;0;0;80m░░\x1b[0m = uncertain");
    println!("  \x1b[38;2;0;0;255m██\x1b[0m = 1.0 (strongly class 1)");

    println!("\nCorner reference:");
    println!("  Bottom-left  [0.0, 0.0]");
    println!("  Bottom-right [1.0, 0.0]");
    println!("  Top-left     [0.0, 1.0]");
    println!("  Top-right    [1.0, 1.0]");

    if verbose {
        // Show some sample predictions at the corners
        println!("\nCorner predictions:");
        let bottom_left = predictions.data[0][0];
        let bottom_right = predictions.data[resolution - 1][0];
        let top_left = predictions.data[(resolution - 1) * resolution][0];
        let top_right = predictions.data[resolution * resolution - 1][0];

        println!("  [0.0, 0.0]: {:.4}", bottom_left);
        println!("  [1.0, 0.0]: {:.4}", bottom_right);
        println!("  [0.0, 1.0]: {:.4}", top_left);
        println!("  [1.0, 1.0]: {:.4}", top_right);
    }
}

