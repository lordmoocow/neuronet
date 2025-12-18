//! Neural network testing functionality

use std::process;

use crate::cli::TestArgs;
use crate::data::load_data;
use crate::model::load_model;

/// Test a trained model on data.
pub fn handle_test(args: TestArgs) {
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
