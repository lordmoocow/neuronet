use clap::Parser;
use textplots::{Chart, ColorPlot, Plot};
use rgb::RGB8;
use std::process;

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

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Train(train_args) => handle_train(train_args),
        Commands::Test(test_args) => handle_test(test_args),
        Commands::Visualise(vis_args) => handle_visualisation(vis_args),
    }
}

fn handle_train(args: cli::TrainArgs) {
    println!("Neural Network Training\n");

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

        // Sample loss and predictions for plotting
        if epoch % args.sample_rate == 0 {
            losses.push(loss);
            predictions.push(prediction);
        }
    }

    eprintln!("100%!");

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
    for i in (0..resolution).rev() {
        for j in 0..resolution {
            let index = i * resolution + j;
            let value = predictions.data[index][0];

            // Map prediction value to ASCII characters
            let symbol = match value {
                v if v < 0.1 => " ",
                v if v < 0.3 => "·",
                v if v < 0.5 => "░",
                v if v < 0.7 => "▒",
                v if v < 0.9 => "▓",
                _ => "█",
            };
            print!("{}", symbol);
        }
        println!();
    }

    println!("\nLegend:");
    println!("  ' ' = 0.0-0.1 (strongly 0)");
    println!("  '·' = 0.1-0.3");
    println!("  '░' = 0.3-0.5 (uncertain)");
    println!("  '▒' = 0.5-0.7");
    println!("  '▓' = 0.7-0.9");
    println!("  '█' = 0.9-1.0 (strongly 1)");

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