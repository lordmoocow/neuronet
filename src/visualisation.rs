//! Terminal visualization for neural network decision boundaries
//!
//! Provides both static visualization of trained models and animated
//! playback of training progression with flicker-free rendering.

use std::io::{self, Write};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::cli::VisualiseArgs;
use crate::export::{self, Metadata};
use crate::matrix::Matrix;
use crate::model::load_model;

// ============================================================================
// Terminal Control
// ============================================================================

/// RAII guard for terminal state during animation.
///
/// On creation, enters alternate screen buffer and hides the cursor.
/// On drop (including panics), restores the cursor and exits alternate screen.
pub struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    /// Enter animation mode: alternate screen + hide cursor
    pub fn new() -> io::Result<Self> {
        let mut stdout = io::stdout();
        // Enter alternate screen buffer + hide cursor
        write!(stdout, "\x1b[?1049h\x1b[?25l")?;
        stdout.flush()?;
        Ok(Self { active: true })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.active {
            let mut stdout = io::stdout();
            // Show cursor + exit alternate screen buffer
            let _ = write!(stdout, "\x1b[?25h\x1b[?1049l");
            let _ = stdout.flush();
        }
    }
}

// ============================================================================
// Grid Generation
// ============================================================================

/// Generate a 2D grid of points in [0,1] x [0,1] for visualization.
///
/// Creates resolution^2 points evenly distributed across the unit square.
/// Used for both training boundary tracking and visualization.
pub fn generate_2d_grid(resolution: usize) -> Matrix {
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

// ============================================================================
// Rendering Helpers
// ============================================================================

/// Convert prediction value to RGB color.
///
/// Red (0.0) -> Black (0.5) -> Blue (1.0) gradient.
fn prediction_to_color(value: f64) -> (u8, u8, u8) {
    if value < 0.5 {
        // Black to Red: red increases with distance from 0.5
        let t = (0.5 - value) * 2.0;
        ((255.0 * t) as u8, 0, 0)
    } else {
        // Black to Blue: blue increases with distance from 0.5
        let t = (value - 0.5) * 2.0;
        (0, 0, (255.0 * t) as u8)
    }
}

/// Get display symbol based on confidence level.
///
/// Higher confidence = denser symbol. Doubled for better aspect ratio.
fn confidence_to_symbol(confidence: f64) -> &'static str {
    match confidence {
        c if c > 0.8 => "██",
        c if c > 0.6 => "▓▓",
        c if c > 0.4 => "▒▒",
        c if c > 0.2 => "░░",
        _ => "  ",
    }
}

/// Render an animated frame to a string buffer (flicker-free).
///
/// Builds the entire frame in memory, then writes it in a single operation.
fn render_animated_frame(
    predictions: &Matrix,
    resolution: usize,
    epoch: usize,
    total_epochs: usize,
    frame_num: usize,
    total_frames: usize,
    metadata: &Metadata,
) -> String {
    // Pre-allocate buffer (approx: resolution^2 * 30 bytes per cell + header)
    let mut buffer = String::with_capacity(resolution * resolution * 30 + 500);

    // Move cursor home (no clear!)
    buffer.push_str("\x1b[H");

    // Header
    buffer.push_str(&format!(
        "Epoch: {:>5} / {}  Frame: {:>4} / {}\n",
        epoch,
        total_epochs,
        frame_num + 1,
        total_frames
    ));
    buffer.push_str(&format!(
        "Seed: {}  LR: {}  Architecture: {}\n",
        metadata.seed, metadata.hyperparameters.learning_rate, metadata.architecture.layers
    ));
    buffer.push('\n');

    // Render grid (reversed so y increases upward)
    for i in (0..resolution).rev() {
        for j in 0..resolution {
            let index = i * resolution + j;
            let value = predictions.data[index][0];
            let confidence = (value - 0.5).abs() * 2.0;
            let (r, g, b) = prediction_to_color(value);
            let symbol = confidence_to_symbol(confidence);
            buffer.push_str(&format!("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, symbol));
        }
        buffer.push('\n');
    }

    // Clear any remaining lines from previous frames
    buffer.push_str("\x1b[J");

    buffer
}

// ============================================================================
// Static Visualization
// ============================================================================

/// Display decision boundary with legend (for static visualization).
pub fn display_decision_boundary(predictions: &Matrix, resolution: usize, verbose: bool) {
    println!("\nDecision Boundary Visualization");
    println!("(Input space from 0.0 to 1.0 in both dimensions)\n");

    if verbose {
        println!(
            "Resolution: {}x{} = {} points",
            resolution, resolution, predictions.rows
        );
        println!();
    }

    // Display the grid (reversed so y increases upward like a normal graph)
    for i in (0..resolution).rev() {
        for j in 0..resolution {
            let index = i * resolution + j;
            let value = predictions.data[index][0];
            let confidence = (value - 0.5).abs() * 2.0;
            let (r, g, b) = prediction_to_color(value);
            let symbol = confidence_to_symbol(confidence);
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

// ============================================================================
// Command Handlers
// ============================================================================

/// Main entry point for visualisation command.
pub fn handle_visualisation(args: VisualiseArgs) {
    match (&args.model, &args.dir) {
        (Some(model_path), None) => {
            handle_static_visualisation(model_path, args.resolution, args.verbose);
        }
        (None, Some(dir_path)) => {
            handle_animated_visualisation(dir_path, &args);
        }
        _ => {
            eprintln!("Error: Must specify either --model or --dir");
            process::exit(1);
        }
    }
}

/// Static visualization of a single model.
fn handle_static_visualisation(model_path: &str, resolution: usize, verbose: bool) {
    if verbose {
        println!("Neural Network 2D Visualisation\n");
    }

    if verbose {
        println!("Loading model from '{}'...", model_path);
    }

    let mut network = match load_model(model_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Error loading model from '{}': {}", model_path, e);
            process::exit(1);
        }
    };

    if verbose {
        println!("Model loaded successfully!");
    }

    let inputs = generate_2d_grid(resolution);

    if verbose {
        println!(
            "Generated {} test samples with {} features\n",
            inputs.rows, inputs.cols
        );
    }

    let predictions = network.predict(&inputs);
    display_decision_boundary(&predictions, resolution, verbose);
}

/// Restore terminal state (show cursor, exit alternate screen).
fn restore_terminal() {
    let mut stdout = io::stdout();
    let _ = write!(stdout, "\x1b[?25h\x1b[?1049l");
    let _ = stdout.flush();
}

/// Animated visualization of training progression (flicker-free).
fn handle_animated_visualisation(dir: &str, args: &VisualiseArgs) {
    // Load metadata
    let metadata = match export::load_metadata(dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading metadata from '{}': {}", dir, e);
            process::exit(1);
        }
    };

    if !metadata.tracking.boundary {
        eprintln!(
            "Error: No boundary data in '{}' (was not tracked during training)",
            dir
        );
        process::exit(1);
    }

    // List boundary files
    let files = match export::list_boundary_files(dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error listing boundary files: {}", e);
            process::exit(1);
        }
    };

    if files.is_empty() {
        eprintln!("Error: No boundary files found in '{}/boundary'", dir);
        process::exit(1);
    }

    // Filter by epoch range if specified
    let files: Vec<_> = files
        .into_iter()
        .filter(|(epoch, _)| {
            args.start_epoch.map_or(true, |s| *epoch >= s)
                && args.end_epoch.map_or(true, |e| *epoch <= e)
        })
        .collect();

    let resolution = metadata.tracking.boundary_resolution;
    let total_epochs = metadata.hyperparameters.epochs;
    let delay = Duration::from_millis(args.delay);
    let total_frames = files.len();

    if args.verbose {
        eprintln!(
            "Animating {} boundary snapshots at {}ms delay",
            total_frames, args.delay
        );
        eprintln!("Resolution: {}x{}", resolution, resolution);
    }

    // Set up Ctrl+C handler to restore terminal on interrupt
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    if ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        restore_terminal();
        process::exit(0);
    })
    .is_err()
    {
        eprintln!("Warning: Could not set Ctrl+C handler");
    }

    // Enter animation mode (alternate screen + hide cursor)
    // TerminalGuard ensures cleanup even on panic via Drop
    let _guard = match TerminalGuard::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error initializing terminal: {}", e);
            process::exit(1);
        }
    };

    let mut stdout = io::stdout();

    'outer: loop {
        for (frame_num, (epoch, path)) in files.iter().enumerate() {
            // Check if interrupted
            if !running.load(Ordering::SeqCst) {
                break 'outer;
            }

            // Load boundary data
            let predictions = match export::load_boundary(path) {
                Ok(p) => p,
                Err(_) => {
                    // Skip frame on error
                    continue;
                }
            };

            // Render frame to buffer
            let frame = render_animated_frame(
                &predictions,
                resolution,
                *epoch,
                total_epochs,
                frame_num,
                total_frames,
                &metadata,
            );

            // Single write for entire frame (flicker-free)
            if stdout.write_all(frame.as_bytes()).is_err() {
                break 'outer;
            }
            if stdout.flush().is_err() {
                break 'outer;
            }

            thread::sleep(delay);
        }

        if !args.loop_animation {
            break;
        }
    }

    // TerminalGuard::drop() automatically restores terminal state
}
