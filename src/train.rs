//! Neural network training functionality

use std::path::Path;
use std::process;

use rand::SeedableRng;

use crate::activation::{Activation, ReLU, Sigmoid};
use crate::cli::{parse_layers, parse_track, TrainArgs};
use crate::data::{load_data, load_targets};
use crate::export;
use crate::layer::Layer;
use crate::loss::MSE;
use crate::model::save_model;
use crate::network::Network;
use crate::visualisation;

/// Compute the output path for a given iteration.
///
/// For single iterations, returns the base path unchanged.
/// For multiple iterations, adds iteration suffix:
/// - Files: "model.json" → "model_001.json"
/// - Directories: "./exp" → "./exp/iter_001"
fn iteration_path(base: &str, iteration: usize, total: usize, is_dir: bool) -> String {
    if total == 1 {
        base.to_string()
    } else if is_dir {
        format!("{}/iter_{:03}", base, iteration)
    } else {
        // Split into stem and extension
        let path = Path::new(base);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(base);
        let ext = path.extension().and_then(|s| s.to_str());
        let parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");

        match ext {
            Some(e) => {
                if parent.is_empty() {
                    format!("{}_{:03}.{}", stem, iteration, e)
                } else {
                    format!("{}/{}_{:03}.{}", parent, stem, iteration, e)
                }
            }
            None => {
                if parent.is_empty() {
                    format!("{}_{:03}", stem, iteration)
                } else {
                    format!("{}/{}_{:03}", parent, stem, iteration)
                }
            }
        }
    }
}

/// Train a neural network from data.
pub fn handle_train(args: TrainArgs) {
    // Generate base seed for all iterations
    let base_seed = args.seed.unwrap_or_else(rand::random::<u64>);

    // Parse track configuration
    let track_config = match parse_track(&args.track) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    // Load data (once for all iterations)
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

    // Validate boundary tracking requires 2D input
    if track_config.boundary && inputs.cols != 2 {
        eprintln!(
            "Error: --track boundary requires 2D input data (got {} features)",
            inputs.cols
        );
        process::exit(1);
    }

    if args.verbose {
        println!(
            "Loaded {} samples with {} features and {} targets",
            inputs.rows, inputs.cols, targets.cols
        );
    }

    // Parse layer specification (once for all iterations)
    let layer_specs = match parse_layers(&args.layers) {
        Ok(specs) => specs,
        Err(e) => {
            eprintln!("Error parsing layers: {}", e);
            process::exit(1);
        }
    };

    // Validate output layer matches target dimensions
    let final_layer_size = layer_specs.last().map(|s| s.size).unwrap_or(inputs.cols);
    if final_layer_size != targets.cols {
        eprintln!(
            "Error: Last layer size ({}) doesn't match target dimensions ({})",
            final_layer_size, targets.cols
        );
        process::exit(1);
    }

    println!(
        "Network architecture: {} -> {} layers -> {}",
        inputs.cols,
        layer_specs.len(),
        targets.cols
    );

    // Training hyperparameters
    let loss_fn = MSE;
    let checkpoint_rate = args.checkpoint_rate.unwrap_or(args.sample_rate * 100);

    // Run training iterations
    for iteration in 1..=args.iterations {
        let seed = base_seed + (iteration as u64 - 1);

        // Compute iteration-specific paths
        let save_path = match (&args.save, &args.output_dir) {
            (None, None) => {
                eprintln!("Error: Must specify --save or --output-dir");
                process::exit(1);
            }
            (Some(path), None) => iteration_path(path, iteration, args.iterations, false),
            (_, Some(dir)) => {
                let iter_dir = iteration_path(dir, iteration, args.iterations, true);
                format!("{}/model.json", iter_dir)
            }
        };

        let iter_output_dir = args
            .output_dir
            .as_ref()
            .map(|dir| iteration_path(dir, iteration, args.iterations, true));

        // Create seeded RNG for this iteration
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Build the neural network
        let mut network = Network::new();
        let mut input_size = inputs.cols;

        for spec in &layer_specs {
            let activation: Box<dyn Activation> = match spec.activation.as_str() {
                "sigmoid" => Box::new(Sigmoid),
                "relu" => Box::new(ReLU),
                _ => unreachable!("Activation already validated"),
            };
            network.add_layer(Layer::new(input_size, spec.size, activation, &mut rng));
            input_size = spec.size;
        }

        // Initialize training output if directory mode
        let mut training_output = if let Some(ref dir) = iter_output_dir {
            Some(
                match export::TrainingOutput::new(
                    dir,
                    seed,
                    &args.data,
                    inputs.cols,
                    targets.cols,
                    &args.layers,
                    args.learning_rate,
                    args.epochs,
                    args.sample_rate,
                    &track_config,
                    args.boundary_resolution,
                    checkpoint_rate,
                    inputs.rows,
                ) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("Error initializing output directory: {}", e);
                        process::exit(1);
                    }
                },
            )
        } else {
            None
        };

        // Pre-generate boundary grid if tracking
        let boundary_grid = training_output
            .as_ref()
            .filter(|o| o.tracking_boundary())
            .map(|o| visualisation::generate_2d_grid(o.boundary_resolution()));

        // Training loop
        if args.iterations > 1 {
            eprint!(
                "Training (iteration {}/{}, seed {}) 0%",
                iteration, args.iterations, seed
            );
        } else {
            eprint!("Training (seed {}) 0%", seed);
        }

        let mut final_loss = 0.0;

        for epoch in 0..args.epochs {
            // Execute training step
            let (loss, prediction) =
                network.train_batch(&inputs, &targets, args.learning_rate, &loss_fn);
            final_loss = loss;

            // Progress indicator
            if epoch % (args.epochs / 33).max(1) == 0 && epoch != 0 {
                eprint!(".");
            }
            if epoch % (args.epochs / 10).max(1) == 0 && epoch != 0 {
                eprint!("{}%", (epoch * 100) / args.epochs);
            }

            // Stream metrics at sample rate
            if let Some(ref mut output) = training_output {
                if epoch % args.sample_rate == 0 || epoch == args.epochs - 1 {
                    let preds: Vec<f64> = prediction.data.iter().map(|row| row[0]).collect();
                    if let Err(e) = output.write_epoch(epoch, loss, &preds) {
                        eprintln!("Error writing metrics: {}", e);
                        process::exit(1);
                    }

                    // Boundary snapshot
                    if let Some(ref grid) = boundary_grid {
                        let boundary_preds = network.predict(grid);
                        if let Err(e) = output.write_boundary(epoch, grid, &boundary_preds) {
                            eprintln!("Error writing boundary: {}", e);
                            process::exit(1);
                        }
                    }
                }

                // Model checkpoint (less frequent)
                if output.tracking_checkpoint()
                    && epoch % output.checkpoint_rate() == 0
                    && epoch != 0
                {
                    if let Err(e) = output.write_checkpoint(epoch, &network) {
                        eprintln!("Error writing checkpoint: {}", e);
                        process::exit(1);
                    }
                }
            }
        }

        eprintln!("100%!");

        // Finalize output directory
        if let Some(output) = training_output {
            match output.finish(final_loss, args.epochs) {
                Ok(dir) => {
                    if args.verbose {
                        println!("Training output written to '{}'", dir)
                    }
                }
                Err(e) => {
                    eprintln!("Error finalizing output: {}", e);
                    process::exit(1);
                }
            }
        }

        if args.verbose {
            // After training, show final predictions
            println!("\nFinal Predictions:");
            let final_predictions = network.predict(&inputs);

            // Print each input/target/prediction
            for i in 0..inputs.rows {
                let input = &inputs.data[i];
                let target_vec = &targets.data[i];
                let prediction_vec = &final_predictions.data[i];

                // Compact format for single-output problems
                if target_vec.len() == 1 {
                    println!(
                        "Input: {:?} → Target: {:.1}, Prediction: {:.4}",
                        input, target_vec[0], prediction_vec[0]
                    );
                } else {
                    println!("Input: {:?}", input);
                    println!("  Target:     {:?}", target_vec);
                    println!("  Prediction: {:?}", prediction_vec);
                }
            }
        }

        // Save the trained model
        if args.verbose {
            println!("\nSaving model to '{}'...", save_path);
        }
        if let Err(e) = save_model(&network, &save_path, inputs.cols, targets.cols, seed) {
            eprintln!("Error saving model: {}", e);
            process::exit(1);
        }

        if args.verbose {
            println!("Model saved successfully!");
            if iteration == args.iterations {
                println!("Training complete!");
            }
        }
    }
}
