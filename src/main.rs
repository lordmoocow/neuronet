use clap::Parser;
use std::process;
use rand::SeedableRng;

use crate::{
    activation::{ReLU, Sigmoid},
    cli::{Cli, Commands, parse_layers, parse_track},
    data::{load_data, load_targets},
    layer::Layer,
    loss::MSE,
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
mod visualisation;

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Train(train_args) => handle_train(train_args),
        Commands::Test(test_args) => handle_test(test_args),
        Commands::Visualise(vis_args) => visualisation::handle_visualisation(vis_args),
    }
}

fn handle_train(args: cli::TrainArgs) {
    // Generate or use provided seed for reproducible weight initialization
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);
    println!("Neural Network Training (seed: {})\n", seed);

    // Create seeded RNG for weight initialization
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Validate output configuration
    let save_path = match (&args.save, &args.output_dir) {
        (None, None) => {
            eprintln!("Error: Must specify --save or --output-dir");
            process::exit(1);
        }
        (Some(path), None) => path.clone(),
        (_, Some(dir)) => format!("{}/model.json", dir),
    };

    // Parse track configuration
    let track_config = match parse_track(&args.track) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

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
        println!(
            "Network architecture: {} -> {} layers -> {}",
            inputs.cols,
            layer_specs.len(),
            targets.cols
        );
    }

    // Training hyperparameters
    let loss_fn = MSE;

    // Initialize training output if directory mode
    let checkpoint_rate = args.checkpoint_rate.unwrap_or(args.sample_rate * 100);
    let mut training_output = if let Some(ref dir) = args.output_dir {
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
    eprint!("Training...0%");
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
            Ok(dir) => println!("Training output written to '{}'", dir),
            Err(e) => {
                eprintln!("Error finalizing output: {}", e);
                process::exit(1);
            }
        }
    }

    // After training, show final predictions
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
    println!("\nSaving model to '{}'...", save_path);
    if let Err(e) = save_model(&network, &save_path, inputs.cols, targets.cols, seed) {
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
