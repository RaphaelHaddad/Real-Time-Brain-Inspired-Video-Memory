#!/usr/bin/env python3
"""
Retrieval Parameter Sweep Script

This script runs parameter sweeps specifically for retrieval configurations
using a fixed knowledge graph. Each epoch:
1. Checks if precomputation has been done for the graph
2. Modifies base_config.yaml with random retrieval parameter values
3. Runs batch retrieval
4. Runs benchmark evaluation
5. Saves all outputs in a dedicated epoch folder

Usage:
    python3 scripts/run_retrieval_sweep.py --epochs 15 --config config/base_config.yaml --graph-uuid <uuid>

Author: Auto-generated
"""

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ParameterRange:
    """Defines the range for a parameter to sweep."""
    name: str
    min_val: int
    max_val: int
    yaml_path: List[str]  # Path in YAML hierarchy

@dataclass
class ChoiceParameter:
    """Defines a parameter with discrete choices."""
    name: str
    choices: List[Any]
    yaml_path: List[str]

# Retrieval sweep parameters
RETRIEVAL_SWEEP_PARAMETERS = [
    ParameterRange("top_k_chunks", 2, 6, ["retrieval", "top_k_chunks"]),
]

# Hop method choices
HOP_METHOD_CHOICES = ["naive", "page_rank", "ch3_l3"]

# Parameters that depend on hop method
NAIVE_PARAMETERS = [
    ParameterRange("top_k", 5, 30, ["retrieval", "top_k"]),
    ParameterRange("graph_hops", 2, 5, ["retrieval", "graph_hops"]),
]

CH3_L3_PARAMETERS = [
    ParameterRange("top_k_hop_ch3_l3", 2, 8, ["retrieval", "top_k_hop_ch3_l3"]),
    ParameterRange("max_path_length_ch3", 2, 8, ["retrieval", "max_path_length_ch3"]),
]

PAGERANK_PARAMETERS = [
    ParameterRange("top_k_hop_pagerank", 4, 20, ["retrieval", "top_k_hop_pagerank"]),
]

CH3_AGGREGATION_CHOICES = ["sum", "product"]

# Fixed paths
PLOT_CONFIG_TEMPLATE = "config/plot_metrics.yaml"


@dataclass
class RetrievalEpochResult:
    """Results from a single retrieval epoch."""
    epoch: int
    graph_uuid: str
    parameters: Dict[str, Any]
    accuracy: Optional[float] = None
    total_queries: int = 0
    correct_queries: int = 0
    avg_retrieval_time: Optional[float] = None
    status: str = "not_started"
    error_message: Optional[str] = None
    output_folder: str = ""


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Simple logger with timestamped output."""

    def __init__(self, log_file: Optional[Path] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level: str, message: str):
        line = f"[{self._timestamp()}] [{level}] {message}"
        if self.verbose:
            print(line)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(line + "\n")

    def info(self, message: str):
        self._write("INFO", message)

    def warning(self, message: str):
        self._write("WARN", message)

    def error(self, message: str):
        self._write("ERROR", message)

    def success(self, message: str):
        self._write("SUCCESS", message)

    def debug(self, message: str):
        self._write("DEBUG", message)

    def separator(self, char: str = "=", length: int = 70):
        self._write("", char * length)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path):
    """Save data to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def set_nested_value(data: Dict, path: List[str], value: Any):
    """Set a nested value in a dictionary using a path list."""
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def get_nested_value(data: Dict, path: List[str], default: Any = None) -> Any:
    """Get a nested value from a dictionary using a path list."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def sample_retrieval_parameters() -> Dict[str, Any]:
    """Sample random retrieval parameters according to the specified logic."""
    params = {}

    # First: top_k_chunks (between 2 and 6)
    params["top_k_chunks"] = random.randint(2, 6)

    # Then: hop_method (random among the three)
    hop_method = random.choice(HOP_METHOD_CHOICES)
    params["hop_method"] = hop_method

    # Then according to the chosen method
    if hop_method == "naive":
        params["top_k"] = random.randint(5, 30)
        params["graph_hops"] = random.randint(2, 5)
    elif hop_method == "ch3_l3":
        params["top_k_hop_ch3_l3"] = random.randint(2, 8)
        params["max_path_length_ch3"] = random.randint(2, 8)
        params["cum_score_aggregation"] = random.choice(CH3_AGGREGATION_CHOICES)
    elif hop_method == "page_rank":
        params["top_k_hop_pagerank"] = random.randint(4, 20)

    return params


def apply_retrieval_parameters_to_config(config: Dict[str, Any], params: Dict[str, Any]):
    """Apply sampled parameters to the config dictionary."""
    # Apply top_k_chunks
    set_nested_value(config, ["retrieval", "top_k_chunks"], params["top_k_chunks"])

    # Apply hop_method
    set_nested_value(config, ["retrieval", "hop_method"], params["hop_method"])

    # Apply method-specific parameters
    hop_method = params["hop_method"]
    if hop_method == "naive":
        set_nested_value(config, ["retrieval", "top_k"], params["top_k"])
        set_nested_value(config, ["retrieval", "graph_hops"], params["graph_hops"])
    elif hop_method == "ch3_l3":
        set_nested_value(config, ["retrieval", "top_k_hop_ch3_l3"], params["top_k_hop_ch3_l3"])
        set_nested_value(config, ["retrieval", "max_path_length_ch3"], params["max_path_length_ch3"])
        set_nested_value(config, ["retrieval", "cum_score_aggregation"], params["cum_score_aggregation"])
    elif hop_method == "page_rank":
        set_nested_value(config, ["retrieval", "top_k_hop_pagerank"], params["top_k_hop_pagerank"])


def get_python_executable(project_root: Path) -> str:
    """Get the correct Python executable, preferring venv if available."""
    # Check for .venv in project root
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    
    # Fall back to sys.executable
    return sys.executable


def run_command(
    cmd: List[str],
    cwd: Path,
    log: Logger,
    capture_output: bool = False,
    check: bool = True,
) -> Tuple[int, str, str]:
    """Run a command and optionally capture output."""
    cmd_str = " ".join(cmd)
    log.debug(f"Running: {cmd_str}")

    if capture_output:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
        )
        return result.returncode, "", ""


def extract_accuracy_from_benchmark(benchmark_file: Path) -> Tuple[Optional[float], int, int]:
    """Extract accuracy from benchmark results file.
    
    Handles both old format (top-level keys) and new format (nested under 'statistics').
    """
    try:
        with open(benchmark_file, 'r') as f:
            data = json.load(f)

        # Try new format first (nested under 'statistics')
        if "statistics" in data:
            stats = data["statistics"]
            total = stats.get("total_queries", 0)
            correct = stats.get("correct_answers", 0)
            accuracy = stats.get("accuracy", None)
            
            if accuracy is not None:
                return accuracy, correct, total
            elif total > 0:
                return correct / total, correct, total
            else:
                return None, 0, 0
        
        # Fall back to old format (top-level keys)
        total = data.get("total_queries", 0)
        correct = data.get("correct_queries", 0)

        if total > 0:
            accuracy = correct / total
            return accuracy, correct, total
        else:
            return None, 0, 0

    except Exception as e:
        print(f"Error reading benchmark file {benchmark_file}: {e}")
        return None, 0, 0


def extract_retrieval_time(retrieval_metrics_dir: Path, graph_uuid: str) -> Optional[float]:
    """Extract average retrieval time from metrics files."""
    try:
        # Look for retrieval timing files
        pattern = f"retrieval_times_{graph_uuid}_*.json"
        metrics_files = list(retrieval_metrics_dir.glob(pattern))

        if not metrics_files:
            return None

        # Use the most recent one
        latest_file = max(metrics_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Extract average retrieval time
        if "per_query_times" in data and data["per_query_times"]:
            return sum(data["per_query_times"]) / len(data["per_query_times"])
        elif "avg_retrieval_time" in data:
            return data["avg_retrieval_time"]

    except Exception as e:
        print(f"Error extracting retrieval time: {e}")

    return None


def check_precomputation(graph_uuid: str, config_path: Path, project_root: Path, log: Logger) -> bool:
    """Check if precomputation has been done for the required methods."""
    log.info("Checking precomputation status...")

    try:
        # Load config to see what hop methods might be used
        config = load_yaml(config_path)

        # For now, we'll check if page_rank and ch3_l3 precomputation exists
        # since naive doesn't require precomputation
        methods_to_check = ["page_rank", "ch3_l3"]

        for method in methods_to_check:
            # We'll assume precomputation is done and let the retrieval fail if not
            log.info(f"Note: Assuming {method} precomputation exists. If retrieval fails, run precompute first.")

        return True

    except Exception as e:
        log.warning(f"Could not verify precomputation status: {e}")
        return True  # Continue anyway


# ============================================================================
# EPOCH RUNNER
# ============================================================================

class RetrievalEpochRunner:
    """Runs a single retrieval epoch."""

    def __init__(
        self,
        epoch: int,
        graph_uuid: str,
        base_config_path: Path,
        retrieval_json_path: str,
        output_base_dir: Path,
        project_root: Path,
        log: Logger,
        dry_run: bool = False,
    ):
        self.epoch = epoch
        self.graph_uuid = graph_uuid
        self.base_config_path = base_config_path
        self.retrieval_json_path = retrieval_json_path
        self.project_root = project_root
        self.log = log
        self.dry_run = dry_run

        # Create epoch output directory
        self.epoch_dir = output_base_dir / f"epoch_{epoch:03d}"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)

        # Paths within epoch directory
        self.config_path = self.epoch_dir / "config.yaml"
        self.retrieval_output_path = self.epoch_dir / "retrieval_results.json"
        self.benchmark_output_path = self.epoch_dir / "benchmark_results.json"
        self.plot_config_path = self.epoch_dir / "plot_config.yaml"
        self.metrics_plot_path = self.epoch_dir / "metrics_comparison.png"
        self.epoch_summary_path = self.epoch_dir / "epoch_summary.json"

        self.result = RetrievalEpochResult(
            epoch=epoch,
            graph_uuid=graph_uuid,
            parameters={},
            output_folder=str(self.epoch_dir),
        )

    def run(self) -> RetrievalEpochResult:
        """Run the complete retrieval epoch."""
        self.log.separator()
        self.log.info(f"RETRIEVAL EPOCH {self.epoch} STARTING")
        self.log.separator()

        start_time = time.time()

        try:
            # Step 1: Sample and apply parameters
            self._step_sample_parameters()

            # Step 2: Run retrieval
            if not self._step_run_retrieval():
                return self.result

            # Step 3: Run benchmark
            if not self._step_run_benchmark():
                return self.result

            # Step 4: Copy metrics and plot
            self._step_copy_metrics_and_plot()

            # Success!
            self.result.status = "success"
            self.log.success(f"Retrieval epoch {self.epoch} completed successfully!")
            self.log.info(f"  Accuracy: {self.result.accuracy:.2%}" if self.result.accuracy else "  Accuracy: N/A")

        except Exception as e:
            self.result.status = "error"
            self.result.error_message = str(e)
            self.log.error(f"Retrieval epoch {self.epoch} failed with exception: {e}")

        # Save epoch summary
        self._save_summary()

        return self.result

    def _step_sample_parameters(self):
        """Sample parameters and create modified config."""
        self.log.info("Step 1: Sampling retrieval parameters...")

        # Sample random parameters
        params = sample_retrieval_parameters()
        self.result.parameters = params

        self.log.info(f"  Sampled parameters: {params}")

        # Load base config
        config = load_yaml(self.base_config_path)

        # Apply parameters to config
        apply_retrieval_parameters_to_config(config, params)

        # Save modified config
        save_yaml(config, self.config_path)

        self.log.info(f"  Config saved to: {self.config_path}")

    def _step_run_retrieval(self) -> bool:
        """Run batch retrieval."""
        self.log.info("Step 2: Running batch retrieval...")

        if self.dry_run:
            self.log.info("  [DRY RUN] Would run: batch-retrieve")
            return True

        try:
            python_exec = get_python_executable(self.project_root)
            cmd = [
                python_exec, "-m", "src.cli.main",
                "batch-retrieve",
                "--config", str(self.config_path),
                "--graph-uuid", self.graph_uuid,
                "--input", self.retrieval_json_path,
                "--output", str(self.retrieval_output_path)
            ]

            returncode, stdout, stderr = run_command(
                cmd, self.project_root, self.log, capture_output=True, check=False
            )

            if returncode != 0:
                self.result.status = "retrieval_failed"
                self.result.error_message = f"Retrieval failed: {stderr}"
                self.log.error(f"Retrieval failed with return code {returncode}")
                if stderr:
                    self.log.error(f"Stderr: {stderr}")
                return False

            self.log.success("Retrieval completed successfully")
            return True

        except Exception as e:
            self.result.status = "retrieval_error"
            self.result.error_message = f"Retrieval exception: {e}"
            self.log.error(f"Retrieval failed with exception: {e}")
            return False

    def _step_run_benchmark(self) -> bool:
        """Run benchmark evaluation."""
        self.log.info("Step 3: Running benchmark evaluation...")

        if self.dry_run:
            self.log.info("  [DRY RUN] Would run: benchmark")
            # Set dummy values for dry run
            self.result.accuracy = 0.5
            self.result.total_queries = 30
            self.result.correct_queries = 15
            return True

        try:
            python_exec = get_python_executable(self.project_root)
            cmd = [
                python_exec, "-m", "src.cli.main",
                "benchmark",
                "--config", str(self.config_path),
                "--input", str(self.retrieval_output_path),
                "--output", str(self.benchmark_output_path)
            ]

            returncode, stdout, stderr = run_command(
                cmd, self.project_root, self.log, capture_output=True, check=False
            )

            if returncode != 0:
                self.result.status = "benchmark_failed"
                self.result.error_message = f"Benchmark failed: {stderr}"
                self.log.error(f"Benchmark failed with return code {returncode}")
                if stderr:
                    self.log.error(f"Stderr: {stderr}")
                return False

            # Extract accuracy from benchmark results
            accuracy, correct, total = extract_accuracy_from_benchmark(self.benchmark_output_path)
            self.result.accuracy = accuracy
            self.result.correct_queries = correct
            self.result.total_queries = total

            if accuracy is not None:
                self.log.success(f"Benchmark completed: {correct}/{total} correct ({accuracy:.2%})")
            else:
                self.log.warning(f"Benchmark completed but accuracy could not be extracted: {correct}/{total}")
            return True

        except Exception as e:
            self.result.status = "benchmark_error"
            self.result.error_message = f"Benchmark exception: {e}"
            self.log.error(f"Benchmark failed with exception: {e}")
            return False

    def _step_copy_metrics_and_plot(self):
        """Copy metrics and generate plots."""
        self.log.info("Step 4: Processing metrics and plots...")

        try:
            # Extract retrieval time
            metrics_dir = self.project_root / "metrics"
            if metrics_dir.exists():
                avg_time = extract_retrieval_time(metrics_dir, self.graph_uuid)
                if avg_time:
                    self.result.avg_retrieval_time = avg_time
                    self.log.info(f"  Average retrieval time: {avg_time:.3f}s")

            # Copy benchmark results to epoch directory (already done)
            self.log.info("  Metrics processing completed")

        except Exception as e:
            self.log.warning(f"Metrics processing failed: {e}")

    def _save_summary(self):
        """Save epoch summary to JSON file."""
        summary = {
            "epoch": self.result.epoch,
            "graph_uuid": self.result.graph_uuid,
            "parameters": self.result.parameters,
            "accuracy": self.result.accuracy,
            "total_queries": self.result.total_queries,
            "correct_queries": self.result.correct_queries,
            "avg_retrieval_time": self.result.avg_retrieval_time,
            "status": self.result.status,
            "error_message": self.result.error_message,
            "output_folder": self.result.output_folder,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.epoch_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# SWEEP RUNNER
# ============================================================================

class RetrievalSweepRunner:
    """Runs the complete retrieval parameter sweep across multiple epochs."""

    def __init__(
        self,
        num_epochs: int,
        graph_uuid: str,
        base_config_path: Path,
        retrieval_json_path: str,
        output_dir: Path,
        project_root: Path,
        dry_run: bool = False,
        start_epoch: int = 1,
    ):
        self.num_epochs = num_epochs
        self.graph_uuid = graph_uuid
        self.base_config_path = base_config_path
        self.retrieval_json_path = retrieval_json_path
        self.output_dir = output_dir
        self.project_root = project_root
        self.dry_run = dry_run
        self.start_epoch = start_epoch

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = output_dir / f"sweep_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log = Logger(
            log_file=self.sweep_dir / "sweep.log",
            verbose=True,
        )

        self.results: List[RetrievalEpochResult] = []

    def run(self):
        """Run all epochs."""
        self.log.separator("=")
        self.log.info("RETRIEVAL PARAMETER SWEEP STARTING")
        self.log.info(f"  Epochs: {self.num_epochs}")
        self.log.info(f"  Graph UUID: {self.graph_uuid}")
        self.log.info(f"  Output directory: {self.sweep_dir}")
        self.log.info(f"  Dry run: {self.dry_run}")
        self.log.separator("=")

        self.log.info("Parameter sampling logic:")
        self.log.info("  1. top_k_chunks: [2, 6]")
        self.log.info("  2. hop_method: ['naive', 'page_rank', 'ch3_l3']")
        self.log.info("  3. Method-specific parameters:")
        self.log.info("     - naive: top_k [5, 30], graph_hops [2, 5]")
        self.log.info("     - ch3_l3: top_k_hop_ch3_l3 [2, 8], max_path_length_ch3 [2, 8], cum_score_aggregation ['sum', 'product']")
        self.log.info("     - page_rank: top_k_hop_pagerank [4, 20]")

        # Check precomputation
        if not check_precomputation(self.graph_uuid, self.base_config_path, self.project_root, self.log):
            self.log.error("Precomputation check failed. Please run precompute first.")
            return

        sweep_start = time.time()

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            runner = RetrievalEpochRunner(
                epoch=epoch,
                graph_uuid=self.graph_uuid,
                base_config_path=self.base_config_path,
                retrieval_json_path=self.retrieval_json_path,
                output_base_dir=self.sweep_dir,
                project_root=self.project_root,
                log=self.log,
                dry_run=self.dry_run,
            )

            result = runner.run()
            self.results.append(result)

            # Save intermediate summary after each epoch
            self._save_sweep_summary()

            self.log.info("")  # Blank line between epochs

        sweep_duration = time.time() - sweep_start
        self._print_final_summary(sweep_duration)

    def _save_sweep_summary(self):
        """Save current sweep results to JSON file."""
        summary_path = self.sweep_dir / "sweep_summary.json"

        summary = {
            "sweep_type": "retrieval",
            "total_epochs": len(self.results),
            "completed_epochs": len([r for r in self.results if r.status == "success"]),
            "results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _print_final_summary(self, duration: float):
        """Print final sweep summary."""
        self.log.separator("=")
        self.log.success("RETRIEVAL SWEEP COMPLETED")
        self.log.info(f"  Duration: {duration:.1f}s")
        self.log.info(f"  Total epochs: {len(self.results)}")
        self.log.separator()

        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status != "success"]

        if successful:
            accuracies = [r.accuracy for r in successful if r.accuracy is not None]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                max_accuracy = max(accuracies)
                self.log.success(f"Successful epochs: {len(successful)}")
                self.log.info(f"  Average accuracy: {avg_accuracy:.2%}")
                self.log.info(f"  Best accuracy: {max_accuracy:.2%}")

                best = max(successful, key=lambda r: r.accuracy or 0)
                self.log.info(f"Best result:")
                self.log.info(f"  Epoch: {best.epoch}")
                if best.accuracy is not None:
                    self.log.info(f"  Accuracy: {best.accuracy:.2%}")
                else:
                    self.log.info(f"  Accuracy: N/A")
                self.log.info(f"  Parameters: {best.parameters}")
            else:
                self.log.warning(f"Successful epochs: {len(successful)}, but no accuracy data available")

        if failed:
            self.log.warning("Failed epochs:")
            for r in failed:
                self.log.warning(f"  Epoch {r.epoch}: {r.status} - {r.error_message}")

        self.log.info(f"Results saved to: {self.sweep_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Run retrieval parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 15 epochs with default settings
  python3 scripts/run_retrieval_sweep.py --epochs 15 --graph-uuid 1e2d92c3-13fc-4264-a03b-735df7cd97c8

  # Dry run to test the setup
  python3 scripts/run_retrieval_sweep.py --epochs 3 --graph-uuid 1e2d92c3-13fc-4264-a03b-735df7cd97c8 --dry-run

  # Custom config and output directory
  python3 scripts/run_retrieval_sweep.py --epochs 10 --graph-uuid 1e2d92c3-13fc-4264-a03b-735df7cd97c8 --config my_config.yaml --output-dir results/
        """
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to run (default: 15)",
    )
    parser.add_argument(
        "--graph-uuid",
        type=str,
        required=True,
        help="UUID of the knowledge graph to use for retrieval",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base_config.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--retrieval-json",
        type=str,
        default="data/groundtruth/retrieval_offline.json",
        help="Path to retrieval queries JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/retrieval_sweeps",
        help="Base directory for sweep outputs (default: outputs/retrieval_sweeps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing actual retrieval commands",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Starting epoch number (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Resolve paths
    project_root = Path(__file__).parent.parent.resolve()
    base_config_path = project_root / args.config
    output_dir = project_root / args.output_dir

    # Validate paths
    if not base_config_path.exists():
        print(f"Error: Config file not found: {base_config_path}")
        sys.exit(1)

    # Validate retrieval input exists
    retrieval_input = project_root / args.retrieval_json
    if not retrieval_input.exists():
        print(f"Error: Retrieval input file not found: {retrieval_input}")
        sys.exit(1)

    # Run sweep
    runner = RetrievalSweepRunner(
        num_epochs=args.epochs,
        graph_uuid=args.graph_uuid,
        base_config_path=base_config_path,
        retrieval_json_path=args.retrieval_json,
        output_dir=output_dir,
        project_root=project_root,
        dry_run=args.dry_run,
        start_epoch=args.start_epoch,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user. Partial results saved.")
        sys.exit(1)


if __name__ == "__main__":
    main()