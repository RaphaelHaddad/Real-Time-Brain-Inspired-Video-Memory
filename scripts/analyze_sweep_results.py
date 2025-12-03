#!/usr/bin/env python3
"""
Analyze and visualize results from parameter sweep runs.

Usage:
    python3 scripts/analyze_sweep_results.py outputs/sweeps/sweep_20251202_123456/

This script:
1. Loads all epoch summaries from a sweep directory
2. Generates correlation analysis between parameters and accuracy
3. Creates visualizations of parameter impact
4. Outputs a summary report

Author: Auto-generated
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_results(sweep_dir: Path) -> Dict[str, Any]:
    """Load sweep summary and all epoch results."""
    summary_path = sweep_dir / "sweep_summary.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Sweep summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_epoch_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Load individual epoch summaries."""
    results = []
    
    for epoch_dir in sorted(sweep_dir.glob("epoch_*")):
        summary_path = epoch_dir / "epoch_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                results.append(json.load(f))
    
    return results


def compute_statistics(epochs: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from epoch results."""
    successful = [e for e in epochs if e.get('status') == 'success']
    
    if not successful:
        return {'error': 'No successful epochs'}
    
    accuracies = [e['accuracy'] for e in successful if e.get('accuracy') is not None]
    retrieval_times = [e['avg_retrieval_time'] for e in successful if e.get('avg_retrieval_time') is not None]
    
    stats = {
        'total_epochs': len(epochs),
        'successful_epochs': len(successful),
        'failed_epochs': len(epochs) - len(successful),
    }
    
    if accuracies:
        stats['accuracy'] = {
            'min': min(accuracies),
            'max': max(accuracies),
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'median': np.median(accuracies),
        }
    
    if retrieval_times:
        stats['retrieval_time'] = {
            'min': min(retrieval_times),
            'max': max(retrieval_times),
            'mean': np.mean(retrieval_times),
            'std': np.std(retrieval_times),
        }
    
    return stats


def compute_parameter_correlations(epochs: List[Dict]) -> Dict[str, float]:
    """Compute correlation between each parameter and accuracy."""
    successful = [e for e in epochs if e.get('status') == 'success' and e.get('accuracy') is not None]
    
    if len(successful) < 3:
        return {}
    
    accuracies = np.array([e['accuracy'] for e in successful])
    
    correlations = {}
    for param_name in successful[0].get('parameters', {}).keys():
        values = np.array([e['parameters'].get(param_name, 0) for e in successful])
        if len(np.unique(values)) > 1:  # Need variation to compute correlation
            corr = np.corrcoef(values, accuracies)[0, 1]
            correlations[param_name] = corr if not np.isnan(corr) else 0.0
    
    return correlations


def find_best_parameters(epochs: List[Dict]) -> Optional[Dict]:
    """Find the parameters that yielded the best accuracy."""
    successful = [e for e in epochs if e.get('status') == 'success' and e.get('accuracy') is not None]
    
    if not successful:
        return None
    
    best = max(successful, key=lambda e: e['accuracy'])
    return {
        'epoch': best['epoch'],
        'accuracy': best['accuracy'],
        'parameters': best['parameters'],
        'graph_uuid': best['graph_uuid'],
    }


def plot_accuracy_distribution(epochs: List[Dict], output_path: Path):
    """Plot histogram of accuracy distribution."""
    accuracies = [e['accuracy'] for e in epochs if e.get('accuracy') is not None]
    
    if not accuracies:
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=min(10, len(accuracies)), edgecolor='black', alpha=0.7)
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.title('Accuracy Distribution Across Epochs')
    plt.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.2%}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_parameter_vs_accuracy(epochs: List[Dict], output_path: Path):
    """Plot each parameter vs accuracy."""
    successful = [e for e in epochs if e.get('status') == 'success' and e.get('accuracy') is not None]
    
    if len(successful) < 2:
        return
    
    param_names = list(successful[0].get('parameters', {}).keys())
    n_params = len(param_names)
    
    if n_params == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    accuracies = [e['accuracy'] for e in successful]
    
    for i, param_name in enumerate(param_names[:6]):  # Max 6 parameters
        ax = axes[i]
        values = [e['parameters'].get(param_name, 0) for e in successful]
        
        ax.scatter(values, accuracies, alpha=0.6, s=50)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{param_name} vs Accuracy')
        
        # Add trend line
        if len(np.unique(values)) > 1:
            z = np.polyfit(values, accuracies, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(values), max(values), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5)
    
    # Hide unused subplots
    for i in range(n_params, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_over_epochs(epochs: List[Dict], output_path: Path):
    """Plot accuracy progression over epochs."""
    epoch_nums = []
    accuracies = []
    
    for e in epochs:
        if e.get('accuracy') is not None:
            epoch_nums.append(e['epoch'])
            accuracies.append(e['accuracy'])
    
    if not accuracies:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_nums, accuracies, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.axhline(np.mean(accuracies), color='red', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(accuracies):.2%}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(sweep_dir: Path, epochs: List[Dict]) -> str:
    """Generate a markdown report."""
    stats = compute_statistics(epochs)
    correlations = compute_parameter_correlations(epochs)
    best = find_best_parameters(epochs)
    
    lines = [
        "# Parameter Sweep Analysis Report",
        "",
        f"**Sweep Directory:** `{sweep_dir}`",
        "",
        "## Summary Statistics",
        "",
        f"- Total epochs: {stats.get('total_epochs', 0)}",
        f"- Successful: {stats.get('successful_epochs', 0)}",
        f"- Failed: {stats.get('failed_epochs', 0)}",
        "",
    ]
    
    if 'accuracy' in stats:
        acc = stats['accuracy']
        lines.extend([
            "### Accuracy",
            "",
            f"- Min: {acc['min']:.2%}",
            f"- Max: {acc['max']:.2%}",
            f"- Mean: {acc['mean']:.2%}",
            f"- Std: {acc['std']:.4f}",
            f"- Median: {acc['median']:.2%}",
            "",
        ])
    
    if 'retrieval_time' in stats:
        rt = stats['retrieval_time']
        lines.extend([
            "### Retrieval Time",
            "",
            f"- Min: {rt['min']:.2f}s",
            f"- Max: {rt['max']:.2f}s",
            f"- Mean: {rt['mean']:.2f}s",
            "",
        ])
    
    if correlations:
        lines.extend([
            "## Parameter Correlations with Accuracy",
            "",
            "| Parameter | Correlation |",
            "|-----------|-------------|",
        ])
        for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"| {param} | {corr:+.3f} |")
        lines.append("")
    
    if best:
        lines.extend([
            "## Best Result",
            "",
            f"- **Epoch:** {best['epoch']}",
            f"- **Accuracy:** {best['accuracy']:.2%}",
            f"- **Graph UUID:** `{best['graph_uuid']}`",
            "",
            "### Best Parameters",
            "",
        ])
        for param, value in best['parameters'].items():
            lines.append(f"- {param}: {value}")
        lines.append("")
    
    # Epoch details table
    lines.extend([
        "## Epoch Details",
        "",
        "| Epoch | Status | Accuracy | Retrieval Time |",
        "|-------|--------|----------|----------------|",
    ])
    
    for e in epochs:
        status = "✓" if e.get('status') == 'success' else "✗"
        acc = f"{e['accuracy']:.2%}" if e.get('accuracy') is not None else "N/A"
        rt = f"{e['avg_retrieval_time']:.2f}s" if e.get('avg_retrieval_time') is not None else "N/A"
        lines.append(f"| {e['epoch']} | {status} | {acc} | {rt} |")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter sweep results",
    )
    
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    
    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)
    
    print(f"Analyzing sweep results in: {sweep_dir}")
    
    # Load results
    epochs = load_epoch_results(sweep_dir)
    
    if not epochs:
        print("No epoch results found!")
        sys.exit(1)
    
    print(f"Found {len(epochs)} epoch results")
    
    # Generate plots
    if not args.no_plots:
        print("Generating plots...")
        plot_accuracy_distribution(epochs, sweep_dir / "accuracy_distribution.png")
        plot_parameter_vs_accuracy(epochs, sweep_dir / "parameter_vs_accuracy.png")
        plot_accuracy_over_epochs(epochs, sweep_dir / "accuracy_over_epochs.png")
        print("Plots saved!")
    
    # Generate report
    report = generate_report(sweep_dir, epochs)
    report_path = sweep_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    # Print summary to console
    stats = compute_statistics(epochs)
    best = find_best_parameters(epochs)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Successful epochs: {stats.get('successful_epochs', 0)}/{stats.get('total_epochs', 0)}")
    
    if 'accuracy' in stats:
        print(f"Accuracy: {stats['accuracy']['mean']:.2%} (±{stats['accuracy']['std']:.4f})")
    
    if best:
        print(f"Best accuracy: {best['accuracy']:.2%} (epoch {best['epoch']})")
        print(f"Best parameters: {best['parameters']}")


if __name__ == "__main__":
    main()
