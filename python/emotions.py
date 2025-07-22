import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

EEG_CONFIG = {
    "frontal_pairs": [
        ("F3", "F4"),  # Primary frontal asymmetry
        ("AF3", "AF4"),  # Anterior frontal
        ("F7", "F8"),  # Lateral frontal
    ],
    "frontal_electrodes": ["F3", "F4", "AF3", "AF4", "F7", "F8", "FC5", "FC6"],
    "parietal_pairs": [
        ("P7", "P8"),  # Parietal asymmetry
    ],
    "parietal_electrodes": ["P7", "P8"],
    "frequency_bands": ["alpha", "betaL", "betaH", "theta", "gamma"],
}


def load_eeg_data(filename: str) -> pd.DataFrame:
    """Load EEG data from CSV file"""
    df = pd.read_csv(filename)
    # Remove the first unnamed column if it exists
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(df.columns[0], axis=1)
    return df


def calculate_asymetryies(df: pd.DataFrame, eeg_config=EEG_CONFIG) -> dict:
    """
    Calculate EEG asymmetries for valence, dominance, and activation
    Based on Davidson's model with enhanced normalization and validation
    """
    results = {}
    results["methods"] = ["standard", "normalized", "ratio", "ratio_norm"]

    # Define electrode pairs for asymmetry calculations
    frontal_pairs = eeg_config["frontal_pairs"]
    parietal_pairs = eeg_config["parietal_pairs"]
    frequency_bands = eeg_config["frequency_bands"]

    # Single merged loop for all asymmetry calculations
    for band in frequency_bands:
        # Initialize lists for different calculation methods
        frontal_asymmetries = []
        frontal_asymmetries_norm = []

        frontal_asymmetries_rational = []
        frontal_asymmetries_rational_norm = []

        parietal_asymmetries = []
        parietal_asymmetries_norm = []

        parietal_asymmetries_rational = []
        parietal_asymmetries_rational_norm = []

        # Calculate asymmetries with all methods
        for pair_type, pairs in [
            ("frontal", frontal_pairs),
            ("parietal", parietal_pairs),
        ]:
            for left, right in pairs:
                left_col = f"{left}/{band}"
                right_col = f"{right}/{band}"

                # Calculate total power for normalization (more robust)
                left_total = 0
                right_total = 0

                for freq in frequency_bands:
                    left_total += df[f"{left}/{freq}"]
                    right_total += df[f"{right}/{freq}"]

                # Raw values with epsilon for numerical stability
                left_raw = df[left_col] + 1e-10
                right_raw = df[right_col] + 1e-10

                # Normalized values
                left_norm = df[left_col] / (left_total + 1e-6)
                right_norm = df[right_col] / (right_total + 1e-6)

                # Method 1: Standard log asymmetry
                asymmetry_log = np.log(right_raw) - np.log(left_raw)

                # Method 2: Normalized log asymmetry
                asymmetry_norm_log = np.log(right_norm + 1e-10) - np.log(left_norm + 1e-10)

                # Method 3: Raw ratio asymmetry
                ratio_raw = right_raw / left_raw

                # Method 4: Normalized ratio asymmetry
                ratio_norm = right_norm / (left_norm + 1e-6)

                if pair_type == "frontal":
                    frontal_asymmetries.append(asymmetry_log)
                    frontal_asymmetries_norm.append(asymmetry_norm_log)
                    frontal_asymmetries_rational.append(ratio_raw)
                    frontal_asymmetries_rational_norm.append(ratio_norm)
                else:  # Parietal
                    parietal_asymmetries.append(asymmetry_log)
                    parietal_asymmetries_norm.append(asymmetry_norm_log)
                    parietal_asymmetries_rational.append(ratio_raw)
                    parietal_asymmetries_rational_norm.append(ratio_norm)

        # Store averaged results for each method
        results[f"frontal_asymmetry_{band}"] = np.mean(frontal_asymmetries, axis=0)
        results[f"frontal_asymmetry_norm_{band}"] = np.mean(frontal_asymmetries_norm, axis=0)

        results[f"frontal_asymmetry_rational_{band}"] = np.mean(frontal_asymmetries_rational, axis=0)
        results[f"frontal_asymmetry_rational_norm_{band}"] = np.mean(frontal_asymmetries_rational_norm, axis=0)

        results[f"parietal_asymmetry_{band}"] = np.mean(parietal_asymmetries, axis=0)
        results[f"parietal_asymmetry_norm_{band}"] = np.mean(parietal_asymmetries_norm, axis=0)

        results[f"parietal_asymmetry_rational_{band}"] = np.mean(parietal_asymmetries_rational, axis=0)
        results[f"parietal_asymmetry_rational_norm_{band}"] = np.mean(parietal_asymmetries_rational_norm, axis=0)

    return results


def calculate_valence_dominance_activation(df: pd.DataFrame, eeg_config=EEG_CONFIG) -> dict:
    """
    Calculate valence, dominance, and activation using multiple EEG asymmetry methods
    Based on Davidson's model with enhanced normalization and validation
    """

    results = {}
    asymmetries = calculate_asymetryies(df)

    # Calculate final metrics with multiple approaches
    valence_methods = []
    dominance_methods = []

    # Valence from different alpha asymmetry methods
    valence_methods.append(-asymmetries["frontal_asymmetry_alpha"])  # Standard log
    valence_methods.append(-asymmetries["frontal_asymmetry_norm_alpha"])  # Normalized log

    # For ratio methods, convert to log scale for consistency
    ratio_raw = asymmetries["frontal_asymmetry_rational_alpha"]
    ratio_norm = asymmetries["frontal_asymmetry_rational_norm_alpha"]
    valence_methods.append(-np.log(ratio_raw + 1e-10))  # Ratio as log
    valence_methods.append(-np.log(ratio_norm + 1e-10))

    # Dominance from parietal asymmetries
    dominance_methods.append(asymmetries["parietal_asymmetry_alpha"])
    dominance_methods.append(asymmetries["parietal_asymmetry_norm_alpha"])
    dominance_methods.append(np.log(asymmetries["parietal_asymmetry_rational_alpha"] + 1e-10))
    dominance_methods.append(np.log(asymmetries["parietal_asymmetry_rational_norm_alpha"] + 1e-10))

    # Create composite scores (weighted average)
    if len(valence_methods) == 4:
        # Weight: standard=0.3, normalized=0.3, ratio=0.2, ratio_norm=0.2
        weights = [0.3, 0.3, 0.2, 0.2]
        results["valence"] = np.average(valence_methods, weights=weights, axis=0)
    else:
        print(f"Warning: Expected 4 valence methods, got {len(valence_methods)}")
        results["valence"] = np.mean(valence_methods, axis=0)

    # Store individual methods for comparison
    method_names = asymmetries["methods"]
    for i, method in enumerate(method_names[: len(valence_methods)]):
        results[f"valence_{method}"] = valence_methods[i]

    # Dominance calculation
    results["dominance"] = np.mean(dominance_methods, axis=0)

    # Store individual methods
    for i, method in enumerate(method_names[: len(dominance_methods)]):
        results[f"dominance_{method}"] = dominance_methods[i]

    # Enhanced activation calculation
    frontal_electrodes = eeg_config["frontal_electrodes"]

    # Multiple activation measures
    beta_low_activities = []
    beta_high_activities = []
    beta_combined_activities = []

    for electrode in frontal_electrodes:
        beta_low_col = f"{electrode}/betaL"
        beta_high_col = f"{electrode}/betaH"

        beta_low_activities.append(df[beta_low_col])
        beta_high_activities.append(df[beta_high_col])
        beta_combined_activities.append(df[beta_low_col] + df[beta_high_col])

    # Store different activation measures
    results["activation_beta_low"] = np.mean(beta_low_activities, axis=0)
    results["activation_beta_high"] = np.mean(beta_high_activities, axis=0)
    results["activation_beta_combined"] = np.mean(beta_combined_activities, axis=0)

    # Primary activation measure (combined beta for robustness)
    results["activation_combined"] = results["activation_beta_combined"]
    results["activation"] = results["activation_beta_low"]

    return results


def calculate_hjorth_parameters(df: pd.DataFrame, eeg_config=EEG_CONFIG) -> dict:
    """
    Calculate Hjorth parameters (Activity, Mobility, Complexity) for each electrode
    """
    hjorth_results = {}
    bands = eeg_config["frequency_bands"]

    # Get all unique electrodes
    electrodes = set()
    for col in df.columns:
        if "/" in col:
            electrode = col.split("/")[0]
            electrodes.add(electrode)

    for electrode in electrodes:
        # Get all frequency bands for this electrode
        electrode_data = []

        for band in bands:
            col_name = f"{electrode}/{band}"
            if col_name in df.columns:
                electrode_data.append(df[col_name].values)

        if electrode_data:
            # Combine all frequency bands as a multi-dimensional signal
            signal_data = np.array(electrode_data).T  # Transpose to have time x frequency

            # Calculate Hjorth parameters for the combined signal
            hjorth_params = calculate_hjorth_single_electrode(signal_data, electrode)

            hjorth_results[f"activity"] = hjorth_params[f"{electrode}_activity"]
            hjorth_results[f"mobility"] = hjorth_params[f"{electrode}_mobility"]
            hjorth_results[f"complexity"] = hjorth_params[f"{electrode}_complexity"]

    return hjorth_results


def calculate_hjorth_single_electrode(signal_data: np.ndarray, electrode: str) -> dict:
    """
    Calculate Hjorth parameters for a single electrode
    """
    if signal_data.ndim == 1:
        x = signal_data
    else:
        # For multi-dimensional data, use the first principal component
        x = np.mean(signal_data, axis=1)

    # First derivative (approximated by differences)
    dx = np.diff(x)

    # Second derivative
    ddx = np.diff(dx)

    # Variances
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    # Hjorth parameters
    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0
    complexity = np.sqrt(var_ddx / var_dx) / mobility if var_dx > 0 and mobility > 0 else 0

    return {f"{electrode}_activity": activity, f"{electrode}_mobility": mobility, f"{electrode}_complexity": complexity}


def plot_valence_activation(vda_results, save_plot=True, dominance_is_size=True, bin_size=1):
    """
    Create a scatter plot of valence vs activation with quadrant analysis
    """
    if "valence" not in vda_results or "activation" not in vda_results or "dominance" not in vda_results:
        print("Error: Valence, activation, or dominance data not available for plotting")
        return

    valence = vda_results["valence"]
    activation = vda_results["activation"]
    dominance = vda_results["dominance"]

    n_points = len(valence)
    n_bins = n_points // bin_size

    valence_binned = []
    activation_binned = []
    dominance_binned = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size

        valence_binned.append(np.mean(valence[start_idx:end_idx]))
        activation_binned.append(np.mean(activation[start_idx:end_idx]))
        dominance_binned.append(np.mean(dominance[start_idx:end_idx]))

    # Convert to numpy arrays
    valence = np.array(valence_binned)
    activation = np.array(activation_binned)
    dominance = np.array(dominance_binned)

    print(f"Reduced from {n_points} to {len(valence)} points")

    sizes = 50
    if dominance_is_size:
        dom_min, dom_max = np.min(dominance), np.max(dominance)
        if dom_max != dom_min:  # Avoid division by zero
            sizes = 1 + 100 * (dominance - dom_min) / (dom_max - dom_min)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(
        valence,
        activation,
        alpha=0.7,
        s=sizes,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add quadrant lines
    plt.axhline(y=np.mean(activation), color="red", linestyle="--", alpha=0.6, linewidth=1, label="Data mean")
    plt.axvline(x=np.mean(valence), color="red", linestyle="--", alpha=0.6, linewidth=1)

    # Add quadrant lines at (0,0)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=2, label="Zero line")
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=2)

    # Force (0,0) as the visual center
    # TODO : Use a more method with consistent axis limits
    x_absmax = max(abs(np.min(valence)), abs(np.max(valence)), 0.1) * 1.1
    y_absmax = max(abs(np.min(activation)), abs(np.max(activation)), 0.1) * 1.1
    axis_limit = max(x_absmax, y_absmax)
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)

    # Get current axis limits
    x_range = plt.xlim()
    y_range = plt.ylim()

    plt.text(
        x_range[1] * 0.8,
        y_range[1] * 0.9,
        "High Activation\nPositive Valence\n(Happy/Excited)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[1] * 0.9,
        "High Activation\nNegative Valence\n(Angry/Stressed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.text(
        x_range[1] * 0.8,
        y_range[0] * 0.8,
        "Low Activation\nPositive Valence\n(Calm/Relaxed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[0] * 0.8,
        "Low Activation\nNegative Valence\n(Sad/Depressed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Customize the plot
    plt.xlabel("Valence (Negative ← → Positive)", fontsize=12, fontweight="bold")
    plt.ylabel("Activation (Low ← → High)", fontsize=12, fontweight="bold")
    plt.title("EEG-based Emotional State: Valence vs Activation", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Data Points: {len(valence)}\n"
    stats_text += f"Valence: μ={np.mean(valence):.3f}, σ={np.std(valence):.3f}\n"
    stats_text += f"Activation: μ={np.mean(activation):.3f}, σ={np.std(activation):.3f}\n"
    stats_text += f"Dominance: μ={np.mean(dominance):.3f}, σ={np.std(dominance):.3f}\n"

    # Add quadrant counts
    q1 = np.sum((valence >= 0) & (activation >= 0))  # Happy
    q2 = np.sum((valence < 0) & (activation >= 0))  # Angry
    q3 = np.sum((valence < 0) & (activation < 0))  # Sad
    q4 = np.sum((valence >= 0) & (activation < 0))  # Calm

    stats_text += f"Quadrants: Happy={q1}, Angry={q2}, Sad={q3}, Calm={q4}"

    plt.text(
        0.02,
        0.9,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="Zero line"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Data mean"),
    ]

    if dominance_is_size:
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Size ∝ Dominance")
        )

    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.85))
    # plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))

    plt.tight_layout()

    if save_plot:
        plt.savefig("valence_activation_plot.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'valence_activation_plot.png'")

    plt.show()


def plot_time_series(vda_results, save_plot=True):
    """
    Create time series plots for valence, activation, and dominance
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    metrics = ["valence", "activation", "dominance"]
    colors = ["blue", "red", "green"]
    labels = [
        "Valence (Negative ← → Positive)",
        "Activation (Low ← → High)",
        "Dominance (Submissive ← → Dominant)",
    ]

    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        if metric in vda_results:
            data = vda_results[metric]
            axes[i].plot(data, color=color, linewidth=1.5, alpha=0.8)
            axes[i].axhline(y=np.mean(data), color="black", linestyle="--", alpha=0.5)
            axes[i].set_ylabel(label, fontweight="bold")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"{metric.capitalize()} Over Time", fontweight="bold")

            # Add statistics
            stats_text = f"μ={np.mean(data):.3f}, σ={np.std(data):.3f}"
            axes[i].text(
                0.02,
                0.95,
                stats_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    axes[-1].set_xlabel("Time Points", fontweight="bold")
    plt.suptitle("EEG Emotional Metrics Time Series", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_plot:
        plt.savefig("emotional_metrics_timeseries.png", dpi=300, bbox_inches="tight")
        print("Time series plot saved as 'emotional_metrics_timeseries.png'")

    plt.show()


def main():
    """Main function to process EEG data"""
    filename = rf"sub_data\virtual\pow.csv"

    # Load data
    print("Loading EEG data...")
    df = load_eeg_data(filename)

    if df is None:
        return

    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])}...")  # Show first 5 columns

    # Calculate valence, dominance, and activation
    print("\nCalculating valence, dominance, and activation...")
    vda_results = calculate_valence_dominance_activation(df)

    # Calculate Hjorth parameters
    print("Calculating Hjorth parameters...")
    hjorth_results = calculate_hjorth_parameters(df)

    # Display results
    print("\n" + "=" * 50)
    print("VALENCE, DOMINANCE, AND ACTIVATION RESULTS")
    print("=" * 50)

    for key, values in vda_results.items():
        if isinstance(values, np.ndarray):
            print(f"{key}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std:  {np.std(values):.4f}")
            print(f"  Min:  {np.min(values):.4f}")
            print(f"  Max:  {np.max(values):.4f}")
        else:
            print(f"{key}: {values:.4f}")

    print("\n" + "=" * 50)
    print("HJORTH PARAMETERS RESULTS")
    print("=" * 50)

    # Group by electrode
    electrodes = set()
    for key in hjorth_results.keys():
        electrode = key.split("_")[0]
        electrodes.add(electrode)

    for electrode in sorted(electrodes):
        print(f"\n{electrode}:")
        for param in ["activity", "mobility", "complexity"]:
            key = f"{electrode}_{param}"
            if key in hjorth_results:
                print(f"  {param.capitalize()}: {hjorth_results[key]:.4f}")

    # Save results to CSV
    print("\nSaving results...")

    # Prepare VDA results for saving
    vda_df_data = {}
    for key, values in vda_results.items():
        if isinstance(values, np.ndarray):
            vda_df_data[key] = values
        else:
            vda_df_data[key] = [values] * len(df)

    if vda_df_data:
        vda_df = pd.DataFrame(vda_df_data)
        vda_df.to_csv("valence_dominance_activation_results.csv", index=False)
        print("VDA results saved to 'valence_dominance_activation_results.csv'")

    # Save Hjorth results
    hjorth_df = pd.DataFrame([hjorth_results])
    hjorth_df.to_csv("hjorth_parameters_results.csv", index=False)
    print("Hjorth parameters saved to 'hjorth_parameters_results.csv'")

    print("\nAnalysis complete!")

    plot_valence_activation(vda_results)
    plot_time_series(vda_results)
    # plot_hjorth_parameters(hjorth_results)


if __name__ == "__main__":
    main()
