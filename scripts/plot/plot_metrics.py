"""
Simple plotting utility for batch metrics JSON files.

Usage:
  python3 scripts/plot/plot_metrics.py --config config/plot_metrics.yaml

The YAML config lists the JSON files to compare and a set of metric flags; only metrics set to true will be plotted.

Each subplot corresponds to one metric. Each JSON file produces one curve per subplot (labelled).

The script handles missing/null values by skipping points (matplotlib will break the line).
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import yaml
import math
import matplotlib.pyplot as plt


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metric_series(data: List[Dict[str, Any]], metric: str) -> List[Any]:
    """Extract a series for a top-level metric name from each batch's network_metrics.
    Returns list aligned by batch index. Nulls are converted to math.nan.
    """
    series = []
    for batch in data:
        nm = batch.get('network_metrics', {}) if isinstance(batch, dict) else {}
        v = nm.get(metric)
        if v is None:
            series.append(math.nan)
        else:
            series.append(v)
    return series


def make_plots(config: Dict[str, Any]):
    json_files = config.get('json_files', [])
    if not json_files:
        raise ValueError('No json_files defined in config')

    # load and label JSON files
    datasets = []  # list of (label, path, data)
    for entry in json_files:
        path = Path(entry['path']).expanduser()
        label = entry.get('label', path.name)
        if not path.exists():
            raise FileNotFoundError(f"Metrics JSON not found: {path}")
        data = load_json(path)
        datasets.append({'label': label, 'path': str(path), 'data': data})

    # metrics selection
    metrics_flags = config.get('metrics', {})
    # Default order for plotting if not explicitly set
    default_metrics_order = [
        'node_count', 'relationship_count', 'density', 'avg_degree',
        'diameter_estimate', 'clustering_coefficient', 'pagerank_top10_percent',
        'louvain_communities', 'louvain_modularity', 'label_entropy'
    ]
    metrics_to_plot = [m for m in default_metrics_order if metrics_flags.get(m, False)]
    if not metrics_to_plot:
        raise ValueError('No metrics enabled in config.metrics')

    n = len(metrics_to_plot)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=tuple(config.get('plot', {}).get('figsize', [12, rows*3])), squeeze=False)
    fig.suptitle(config.get('plot', {}).get('title', 'Batch Metrics Comparison'), fontsize=14)

    for idx, metric in enumerate(metrics_to_plot):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        # Plot each dataset on this metric
        for ds in datasets:
            series = extract_metric_series(ds['data'], metric)
            x = list(range(len(series)))
            # Matplotlib will break lines at nan; that's fine
            # Slightly more transparent by default for clearer overlap visualization
            alpha = config.get('plot', {}).get('alpha', 0.4)
            lw = config.get('plot', {}).get('linewidth', 2)
            ax.plot(x, series, marker='o', label=ds['label'], alpha=alpha, linewidth=lw)

        ax.set_title(metric)
        ax.set_xlabel('Batch index')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    # Hide any unused subplots
    total_axes = rows * cols
    for empty_idx in range(n, total_axes):
        r = empty_idx // cols
        c = empty_idx % cols
        axes[r][c].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Compute average total_time per dataset and render as footer text
    avg_texts = []
    for ds in datasets:
        times = [b.get('total_time') for b in ds['data'] if isinstance(b, dict) and b.get('total_time') is not None]
        avg = sum(times) / len(times) if times else float('nan')
        avg_texts.append(f"{ds['label']}: avg total_time={avg:.3f}s")

    footer = " | ".join(avg_texts)

    # Render the averages footer with larger, bold font for visibility
    fig.text(0.5, 0.02, footer, ha='center', fontsize=config.get('plot', {}).get('footer_fontsize', 11), weight='bold')

    out_path = Path(config.get('plot', {}).get('save_path', 'outputs/metrics_comparison.png'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    if config.get('plot', {}).get('show', False):
        plt.show()
    print(f"Saved plot to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/plot_metrics.yaml', help='Path to YAML config')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    make_plots(cfg)
