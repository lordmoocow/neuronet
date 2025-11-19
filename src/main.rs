use clap::Parser;
use textplots::{Chart, ColorPlot, Plot};
use rgb::RGB8;
use std::process;

use crate::{
    activation::{ReLU, Sigmoid},
    cli::{Cli, parse_layers},
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

fn main() {
    // Parse command-line arguments
    let args = Cli::parse();

    if !args.quiet {
        println!("Neural Network Training\n");
    }

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
        network.add_layer(Layer::new(input_size, spec.size, activation));
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
    let mut losses = Vec::new();
    let mut predictions = Vec::new();

    // Training loop
    if !args.quiet {
        eprint!("Training...0%");
    }

    for epoch in 0..args.epochs {
        // Execute training step
        let (loss, prediction) = network.train_batch(&inputs, &targets, args.learning_rate, &loss_fn);

        // Progress indicator
        if !args.quiet {
            if epoch % (args.epochs / 33.max(1)) == 0 && epoch != 0 {
                eprint!(".");
            }
            if epoch % (args.epochs / 10.max(1)) == 0 && epoch != 0 {
                eprint!("{}%", (epoch * 100) / args.epochs);
            }
        }

        // Sample loss and predictions for plotting
        if epoch % args.sample_rate == 0 {
            losses.push(loss);
            predictions.push(prediction);
        }
    }

    if !args.quiet {
        eprintln!("100%!");
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

    // Display plots if not disabled
    if !args.no_plots {
        if args.verbose {
            println!("\nSample rate: every {} epochs", args.sample_rate);
        }
        plot_losses(&losses, args.sample_rate);

        // Only show prediction evolution for small datasets
        if inputs.rows <= 10 {
            plot_predictions(&predictions, &inputs, &targets, args.sample_rate);
        } else if args.verbose {
            println!("\nSkipping prediction evolution plot (too many samples)");
        }
    }

    // Save the trained model
    println!("\nSaving model to '{}'...", args.save);
    if let Err(e) = save_model(&network, &args.save, inputs.cols, targets.cols) {
        eprintln!("Error saving model: {}", e);
        process::exit(1);
    }

    println!("Model saved successfully!");
    if args.verbose {
        println!("Training complete!");
    }
}
    if args.verbose {
        println!("\nTraining complete!");
    }
}

fn plot_predictions(predictions: &[Matrix], inputs: &Matrix, targets: &Matrix, sample_rate: usize) {
    // Define colors for each input pattern (up to 10)
    let colors = [
        RGB8::new(255, 0, 0),     // Red
        RGB8::new(0, 255, 0),     // Green
        RGB8::new(0, 128, 255),   // Blue
        RGB8::new(255, 255, 0),   // Yellow
        RGB8::new(255, 0, 255),   // Magenta
        RGB8::new(0, 255, 255),   // Cyan
        RGB8::new(255, 128, 0),   // Orange
        RGB8::new(128, 0, 255),   // Purple
        RGB8::new(128, 255, 0),   // Lime
        RGB8::new(255, 128, 128), // Pink
    ];

    println!("\nPrediction Evolution:");

    // Print legend for first few samples
    let num_samples = inputs.rows.min(10);
    for i in 0..num_samples {
        let input = &inputs.data[i];
        let target = &targets.data[i];
        let color_name = match i {
            0 => "Red",
            1 => "Green",
            2 => "Blue",
            3 => "Yellow",
            4 => "Magenta",
            5 => "Cyan",
            6 => "Orange",
            7 => "Purple",
            8 => "Lime",
            9 => "Pink",
            _ => "?",
        };

        if target.len() == 1 {
            println!("  {:<8} {:?} (target → {:.1})", format!("{}:", color_name), input, target[0]);
        } else {
            println!("  {:<8} {:?} (target → {:?})", format!("{}:", color_name), input, target);
        }
    }
    println!();

    // Create owned data for each series (one per input row)
    let mut series_data: Vec<Vec<(f32, f32)>> = Vec::new();
    for i in 0..num_samples {
        let series = (0..predictions.len())
            .map(|checkpoint| {
                let prediction = predictions[checkpoint].data[i][0] as f32;
                ((checkpoint * sample_rate) as f32, prediction)
            })
            .collect::<Vec<(f32, f32)>>();
        series_data.push(series);
    }

    // Build chart with all series
    // We need to manually expand this for now due to lifetime constraints
    match num_samples {
        1 => {
            let shape0 = textplots::Shape::Lines(&series_data[0]);
            Chart::new(300, 60, 0.0, ((predictions.len() - 1) * sample_rate) as f32)
                .linecolorplot(&shape0, colors[0])
                .display();
        }
        2 => {
            let shape0 = textplots::Shape::Lines(&series_data[0]);
            let shape1 = textplots::Shape::Lines(&series_data[1]);
            Chart::new(300, 60, 0.0, ((predictions.len() - 1) * sample_rate) as f32)
                .linecolorplot(&shape0, colors[0])
                .linecolorplot(&shape1, colors[1])
                .display();
        }
        3 => {
            let shape0 = textplots::Shape::Lines(&series_data[0]);
            let shape1 = textplots::Shape::Lines(&series_data[1]);
            let shape2 = textplots::Shape::Lines(&series_data[2]);
            Chart::new(300, 60, 0.0, ((predictions.len() - 1) * sample_rate) as f32)
                .linecolorplot(&shape0, colors[0])
                .linecolorplot(&shape1, colors[1])
                .linecolorplot(&shape2, colors[2])
                .display();
        }
        _ => {
            // For4 or more samples, plot first 4 only
            let shape0 = textplots::Shape::Lines(&series_data[0]);
            let shape1 = textplots::Shape::Lines(&series_data[1]);
            let shape2 = textplots::Shape::Lines(&series_data[2]);
            let shape3 = textplots::Shape::Lines(&series_data[3]);
            Chart::new(300, 60, 0.0, ((predictions.len() - 1) * sample_rate) as f32)
                .linecolorplot(&shape0, colors[0])
                .linecolorplot(&shape1, colors[1])
                .linecolorplot(&shape2, colors[2])
                .linecolorplot(&shape3, colors[3])
                .display();
        }
    }
}

fn plot_losses(losses: &[f64], sample_rate: usize) {
    println!("\nLoss evolution:");
    Chart::new(300, 60, 0.0, ((losses.len() - 1) * sample_rate) as f32)
        .lineplot(&textplots::Shape::Lines(
            &(0..losses.len())
                .map(|x| ((x * sample_rate) as f32, losses[x] as f32))
                .collect::<Vec<(f32, f32)>>(),
        ))
        .display();
}