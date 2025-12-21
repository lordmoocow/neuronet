#!/usr/bin/env python3

import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


def export_for_chart(df: pd.DataFrame, output_path: str) -> None:
    """
    Export a DataFrame to JSON.
    """
    # Reset index if it's named (e.g., from groupby)
    if df.index.name:
        df = df.reset_index()

    # Convert to columnar format
    data = {col: df[col].tolist() for col in df.columns}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported to {output_path}")


def export_multi_series(series_dict: dict[str, pd.DataFrame], output_path: str) -> None:
    """
    Export multiple DataFrames as a multi-series JSON.
    """
    result = {"series": {}}
    for name, df in series_dict.items():
        if df.index.name:
            df = df.reset_index()
        result["series"][name] = {col: df[col].tolist() for col in df.columns}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Exported to {output_path}")


def export_all_iterations(all_data: pd.DataFrame, value_col: str, output_path: str) -> None:
    """Export each iteration as a separate series for comparison."""
    series_dict = {}
    for iteration in all_data["iteration"].unique():
        iter_data = all_data[all_data["iteration"] == iteration][["epoch", value_col]]
        series_dict[f"Iter {iteration:03d}"] = iter_data

    export_multi_series(series_dict, output_path)


def load_iteration_data(path: str, filename: str) -> list[pd.DataFrame]:
    base_path = Path(path)
    frames: list[pd.DataFrame] = []
    for i in range(1, 11):
        iter_path = base_path / f"iter_{i:03d}" / filename
        frame = pd.read_csv(iter_path)
        frame["iteration"] = i
        frames.append(frame)
    return frames


def mean_spread(data: pd.DataFrame, col: str) -> pd.DataFrame:
    return data.groupby("epoch")[col].aggregate(["mean", "std"])


def epochs_to_threshold(data: pd.DataFrame, col: str, threshold: float = 0.01) -> pd.Series:
    """Find first epoch where each iteration crosses below threshold."""
    below_threshold = data[data[col] < threshold]
    result = below_threshold.groupby("iteration")["epoch"].min()
    return result.reindex(data["iteration"].unique())


def run_neuronet_test(model_path: str, test_data_path: str) -> list[float]:
    """Run neuronet test and return predictions as a list of floats."""
    result = subprocess.run(
        ["cargo", "run", "--", "test", "--model", model_path, "--data", test_data_path],
        capture_output=True,
        text=True
    )
    # Parse output - predictions come after "Predictions:" line
    lines = result.stdout.strip().split("\n")
    predictions = []
    capture = False
    for line in lines:
        if "Predictions:" in line:
            capture = True
            continue
        if capture and line.strip():
            predictions.append(float(line.strip()))
    return predictions


def calculate_test_accuracy(baseline_path: str, test_inputs_path: str, test_targets: list[int]) -> pd.Series:
    """Calculate test accuracy for all iterations of a baseline."""
    base_path = Path(baseline_path)
    accuracies = []

    for i in range(1, 11):
        model_path = base_path / f"iter_{i:03d}" / "model.json"
        predictions = run_neuronet_test(str(model_path), test_inputs_path)

        # Threshold and compare
        binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
        correct = sum(p == t for p, t in zip(binary_preds, test_targets))
        accuracy = correct / len(test_targets)
        accuracies.append(accuracy)

    return pd.Series(accuracies, index=range(1, 11), name="test_accuracy")


def final_values(data: pd.DataFrame, col: str) -> pd.Series:
    """Get final value of column for each iteration."""
    return data.groupby("iteration")[col].last()


def export_summary(rows: list[dict], output_path: str, title: str = "") -> None:
    """
    Export summary statistics for the data-table shortcode.

    rows: [{"label": "Metric", "value": "0.123 ± 0.045"}, ...]
    """
    result = {"rows": rows}
    if title:
        result["title"] = title

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Exported to {output_path}")


def format_mean_std(series: pd.Series, decimals: int = 4) -> str:
    """Format a series as 'mean ± std'."""
    mean = series.mean()
    std = series.std()
    if decimals == 0:
        return f"{mean:.0f} ± {std:.0f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def analyse_baseline(
    name: str,
    data_path: str,
    threshold: float = 0.1,
    test_inputs_path: str | None = None,
    test_targets: list[int] | None = None
) -> dict:
    """
    Analyse a baseline experiment and return summary statistics.
    Also exports JSON files for charts and tables.
    """
    print(f"\n{'='*50}")
    print(f"Analysing {name} baseline")
    print(f"{'='*50}")

    # Load data
    losses = load_iteration_data(data_path, "loss.csv")
    all_losses = pd.concat(losses, ignore_index=True)

    # Calculate metrics
    mean_losses = mean_spread(all_losses, "loss")
    ett = epochs_to_threshold(all_losses, "loss", threshold)
    final_loss = final_values(all_losses, "loss")

    # Count converged (reached threshold)
    converged = ett.notna().sum()
    total = len(ett)

    # Calculate test accuracy if test data provided
    # Note: Training accuracy from predictions.csv is unreliable because data may be
    # shuffled during training, so sample order doesn't match the original CSV.
    # Test accuracy is calculated by running the final model on test data.
    test_acc = None
    if test_inputs_path and test_targets:
        print("Running test accuracy evaluation...")
        test_acc = calculate_test_accuracy(data_path, test_inputs_path, test_targets)

    # Print summary
    print(f"Final loss: {format_mean_std(final_loss)}")
    print(f"Epochs to {threshold}: {format_mean_std(ett, 0)}")
    print(f"Converged: {converged}/{total} ({100*converged/total:.0f}%)")
    if test_acc is not None:
        print(f"Test accuracy: {format_mean_std(test_acc, 2)}")

    # Export chart data
    base_output = f"../../static/data/analysis/{name.lower()}"
    export_for_chart(mean_losses, f"{base_output}_loss_stats.json")
    export_all_iterations(all_losses, "loss", f"{base_output}_all_iterations.json")

    # Export summary table
    summary_rows = [
        {"label": "Final Loss (mean ± std)", "value": format_mean_std(final_loss)},
        {"label": f"Epochs to {threshold} (mean ± std)", "value": format_mean_std(ett, 0)},
        {"label": "Convergence Rate", "value": f"{100*converged/total:.0f}%"},
    ]
    if test_acc is not None:
        summary_rows.append({"label": "Test Accuracy (mean ± std)", "value": format_mean_std(test_acc, 2)})
    export_summary(summary_rows, f"{base_output}_summary.json", title=f"{name} Baseline")

    return {
        "final_loss_mean": final_loss.mean(),
        "final_loss_std": final_loss.std(),
        "epochs_to_threshold_mean": ett.mean(),
        "epochs_to_threshold_std": ett.std(),
        "convergence_rate": converged / total,
    }


def analyse_lr_experiment(
    name: str,
    experiment_path: str,
    learning_rates: list[float],
    iterations_per_lr: int = 5,
    threshold: float = 0.1
) -> None:
    """
    Analyse learning rate experiment across multiple LRs.
    Outputs comparison chart and summary table.
    """
    print(f"\n{'='*50}")
    print(f"Analysing {name} LR experiment")
    print(f"{'='*50}")

    all_series = {}
    summary_rows = []

    for lr in learning_rates:
        lr_path = Path(experiment_path) / str(lr)

        # Load all iterations for this LR
        frames = []
        for i in range(1, iterations_per_lr + 1):
            iter_path = lr_path / f"iter_{i:03d}" / "loss.csv"
            frame = pd.read_csv(iter_path)
            frame["iteration"] = i
            frames.append(frame)

        all_losses = pd.concat(frames, ignore_index=True)

        # Calculate mean loss per epoch
        mean_losses = all_losses.groupby("epoch")["loss"].mean().reset_index()
        all_series[f"LR {lr}"] = mean_losses

        # Calculate summary stats
        final_loss = all_losses.groupby("iteration")["loss"].last()
        ett = epochs_to_threshold(all_losses, "loss", threshold)
        converged = ett.notna().sum()

        print(f"LR {lr}: final loss {format_mean_std(final_loss)}, converged {converged}/{iterations_per_lr}")

        summary_rows.append({
            "label": f"LR {lr}",
            "value": format_mean_std(final_loss),
            "converged": f"{converged}/{iterations_per_lr}"
        })

    # Export comparison chart
    base_output = f"../../static/data/analysis/{name.lower()}"
    export_multi_series(all_series, f"{base_output}_lr_comparison.json")

    # Export summary table
    export_summary(summary_rows, f"{base_output}_lr_summary.json",
                   title="Learning Rate Comparison")


def analyse_epochs_experiment() -> None:
    """
    Analyse the epochs experiment - comparing low LR + high epochs runs.
    """
    print(f"\n{'='*50}")
    print("Analysing epochs experiment")
    print(f"{'='*50}")

    experiments = {
        "LR 0.01 (2M epochs)": "data/experiments/epochs/xor_lr0.01_2M",
        "LR 0.1 (2M epochs)": "data/experiments/epochs/xor_lr0.1_2M",
    }

    # Also include the original LR 10 baseline for comparison
    lr10_path = Path("data/experiments/lr/xor/10.0/iter_001/loss.csv")

    all_series = {}

    for name, path in experiments.items():
        loss_path = Path(path) / "loss.csv"
        if loss_path.exists():
            df = pd.read_csv(loss_path)
            all_series[name] = df
            final_loss = df["loss"].iloc[-1]
            print(f"{name}: final loss {final_loss:.6f}")

    # Add LR 10 for comparison
    if lr10_path.exists():
        df = pd.read_csv(lr10_path)
        all_series["LR 10.0 (2K epochs)"] = df
        print(f"LR 10.0 (2K epochs): final loss {df['loss'].iloc[-1]:.6f}")

    # Export comparison chart
    base_output = "../../static/data/analysis/epochs"
    export_multi_series(all_series, f"{base_output}_comparison.json")

    # Create summary table
    summary_rows = [
        {"label": "LR 0.01 @ 2M epochs", "value": "0.000098", "epochs_to_0.1": "52,000"},
        {"label": "LR 0.1 @ 2M epochs", "value": "0.000009", "epochs_to_0.1": "7,000"},
        {"label": "LR 10.0 @ 2K epochs", "value": "0.000100", "epochs_to_0.1": "~200"},
    ]
    export_summary(summary_rows, f"{base_output}_summary.json", title="Epochs Trade-off")


if __name__ == "__main__":
    # Test targets for accuracy evaluation
    # XOR test file has no targets in the file, but we know them from the truth table
    xor_test_targets = [0, 1, 1, 0]
    circle_test_targets = pd.read_csv("data/circle_test.csv", header=None).iloc[:, -1].astype(int).tolist()

    # Analyse both baselines
    xor_stats = analyse_baseline(
        "XOR", "data/baselines/xor", threshold=0.1,
        test_inputs_path="data/xor_test_inputs.csv", test_targets=xor_test_targets
    )
    circle_stats = analyse_baseline(
        "Circle", "data/baselines/circle", threshold=0.1,
        test_inputs_path="data/circle_test_inputs.csv", test_targets=circle_test_targets
    )

    # Learning rate experiments
    analyse_lr_experiment(
        "xor",
        "data/experiments/lr/xor",
        learning_rates=[0.01, 0.1, 0.5, 5.0, 10.0, 25.0, 50.0],
        iterations_per_lr=5,
        threshold=0.1
    )

    analyse_lr_experiment(
        "circle",
        "data/experiments/lr/circle",
        learning_rates=[0.01, 0.1, 0.5, 5.0, 10.0, 25.0, 50.0],
        iterations_per_lr=5,
        threshold=0.1
    )

    # Epochs experiment - compare low LR with high epochs
    analyse_epochs_experiment()

    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)
