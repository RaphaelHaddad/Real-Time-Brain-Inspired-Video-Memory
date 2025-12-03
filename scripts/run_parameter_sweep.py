#!/usr/bin/env python3
"""
Parameter Sweep Automation Script for KG Pipeline

This script automates running multiple epochs of the KG pipeline with
randomly sampled parameters. Each epoch:
1. Modifies base_config.yaml with random parameter values
2. Builds the KG graph
3. Runs batch retrieval
4. Runs benchmark evaluation
5. Plots metrics comparison (MVP vs current run)
6. Saves all outputs in a dedicated epoch folder

Usage:
    python3 scripts/run_parameter_sweep.py --epochs 15 --config config/base_config.yaml

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
    yaml_path: List[str]  # Path in YAML hierarchy, e.g., ['chunking', 'max_new_triplets']


# Parameters to sweep with their ranges
SWEEP_PARAMETERS = [
    ParameterRange("max_connection_subgraph", 2, 3, ["chunking", "max_connection_subgraph"]),
    ParameterRange("max_new_triplets", 3, 25, ["chunking", "max_new_triplets"]),  # Reduced from 25
    ParameterRange("max_inter_chunk_relations", 1, 20, ["chunking", "max_inter_chunk_relations"]),  # Reduced from 20
    ParameterRange("max_merge_instructions", 1, 15, ["chunking", "max_merge_instructions"]),  # Reduced from 12
    ParameterRange("max_prune_instructions", 1, 20, ["chunking", "max_prune_instructions"]),  # Reduced from 20
]

# Fixed paths
VLM_OUTPUT_PATH = "data/outputs/vlm_output.json"
RETRIEVAL_INPUT_PATH = "data/groundtruth/retrieval_offline.json"
MVP_METRICS_PATH = "data/metrics/mvp_93e9c82e-95d6-4864-8ac1-2ae70edfd961.json"
PLOT_CONFIG_TEMPLATE = "config/plot_metrics.yaml"


@dataclass
class EpochResult:
    """Results from a single epoch."""
    epoch: int
    graph_uuid: str
    parameters: Dict[str, int]
    accuracy: Optional[float] = None
    total_queries: int = 0
    correct_queries: int = 0
    avg_retrieval_time: Optional[float] = None
    kg_build_time: Optional[float] = None
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
# YAML CONFIGURATION HANDLING
# ============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path):
    """Save data to a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def set_nested_value(data: Dict, path: List[str], value: Any):
    """Set a value in a nested dictionary using a path list."""
    for key in path[:-1]:
        data = data.setdefault(key, {})
    data[path[-1]] = value


def get_nested_value(data: Dict, path: List[str], default: Any = None) -> Any:
    """Get a value from a nested dictionary using a path list."""
    for key in path:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


def sample_parameters() -> Dict[str, int]:
    """Sample random values for all sweep parameters."""
    params = {}
    for param in SWEEP_PARAMETERS:
        params[param.name] = random.randint(param.min_val, param.max_val)
    return params


def apply_parameters_to_config(config: Dict[str, Any], params: Dict[str, int]):
    """Apply sampled parameters to a config dictionary."""
    for param in SWEEP_PARAMETERS:
        if param.name in params:
            set_nested_value(config, param.yaml_path, params[param.name])


# ============================================================================
# COMMAND EXECUTION
# ============================================================================

def run_command(
    cmd: List[str],
    log: Logger,
    cwd: Optional[Path] = None,
    capture_output: bool = True,
    log_file: Optional[Path] = None,
    timeout: int = 3600,  # 1 hour default timeout
) -> Tuple[int, str, str]:
    """
    Run a command and return (returncode, stdout, stderr).
    
    If log_file is provided, streams output to that file.
    """
    log.info(f"Running: {' '.join(cmd)}")
    
    try:
        if log_file:
            # Stream output to file
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                stdout_lines = []
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    stdout_lines.append(line)
                process.wait(timeout=timeout)
                return process.returncode, ''.join(stdout_lines), ''
        else:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        log.error(f"Command timed out after {timeout}s")
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        log.error(f"Command failed with exception: {e}")
        return -1, "", str(e)


def extract_uuid_from_output(output: str) -> Optional[str]:
    """Extract the graph UUID from command output."""
    # Pattern: "Knowledge graph construction completed with UUID: <uuid>"
    # Or: "with UUID: <uuid>"
    patterns = [
        r"completed with UUID:\s*([a-f0-9\-]+)",
        r"UUID:\s*([a-f0-9\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_accuracy_from_benchmark(benchmark_file: Path) -> Tuple[Optional[float], int, int]:
    """Extract accuracy from benchmark results JSON."""
    try:
        with open(benchmark_file, 'r') as f:
            data = json.load(f)
        
        stats = data.get('statistics', {})
        accuracy = stats.get('accuracy')
        total = stats.get('total', 0)
        correct = stats.get('correct', 0)
        
        return accuracy, total, correct
    except Exception:
        return None, 0, 0


def extract_retrieval_time(retrieval_metrics_dir: Path, graph_uuid: str) -> Optional[float]:
    """Extract average retrieval time from metrics file."""
    # Look for file matching pattern: retrieval_times_<uuid>_*.json
    for f in retrieval_metrics_dir.glob(f"retrieval_times_{graph_uuid}_*.json"):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            return data.get('average_time')
        except Exception:
            continue
    return None


# ============================================================================
# PLOT CONFIG GENERATION
# ============================================================================

def generate_plot_config(
    mvp_metrics_path: str,
    current_metrics_path: str,
    output_path: str,
    epoch: int,
) -> Dict[str, Any]:
    """Generate a plot_metrics.yaml config for comparison."""
    return {
        'json_files': [
            {'path': current_metrics_path, 'label': f'epoch_{epoch}_subgraph'},
            {'path': mvp_metrics_path, 'label': 'mvp_baseline'},
        ],
        'metrics': {
            'node_count': True,
            'relationship_count': True,
            'density': True,
            'avg_degree': True,
            'avg_unique_neighbors': True,
            'global_efficiency': True,
            'avg_path_length': True,
            'avg_degree_centrality': True,
            'max_degree_centrality': True,
            'avg_betweenness_centrality': True,
            'max_betweenness_centrality': True,
            'degree_assortativity': True,
            'graph_robustness': True,
            'diameter_estimate': True,
            'clustering_coefficient': True,
            'pagerank_top10_percent': True,
            'louvain_communities': True,
            'louvain_modularity': True,
            'label_entropy': True,
        },
        'plot': {
            'title': f'Network Metrics: Epoch {epoch} vs MVP Baseline',
            'figsize': [12, 21],
            'save_path': output_path,
            'show': False,
        }
    }


# ============================================================================
# EPOCH RUNNER
# ============================================================================

class EpochRunner:
    """Runs a single epoch of the pipeline."""
    
    def __init__(
        self,
        epoch: int,
        base_config_path: Path,
        output_base_dir: Path,
        project_root: Path,
        log: Logger,
        dry_run: bool = False,
    ):
        self.epoch = epoch
        self.base_config_path = base_config_path
        self.project_root = project_root
        self.log = log
        self.dry_run = dry_run
        
        # Create epoch output directory
        self.epoch_dir = output_base_dir / f"epoch_{epoch:03d}"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths within epoch directory
        self.config_path = self.epoch_dir / "config.yaml"
        self.kg_log_path = self.epoch_dir / "kg_build.log"
        self.retrieval_output_path = self.epoch_dir / "retrieval_results.json"
        self.benchmark_output_path = self.epoch_dir / "benchmark_results.json"
        self.plot_config_path = self.epoch_dir / "plot_config.yaml"
        self.metrics_plot_path = self.epoch_dir / "metrics_comparison.png"
        self.epoch_summary_path = self.epoch_dir / "epoch_summary.json"
        
        self.result = EpochResult(
            epoch=epoch,
            graph_uuid="",
            parameters={},
            output_folder=str(self.epoch_dir),
        )
    
    def run(self) -> EpochResult:
        """Run the complete epoch pipeline."""
        self.log.separator()
        self.log.info(f"EPOCH {self.epoch} STARTING")
        self.log.separator()
        
        start_time = time.time()
        
        try:
            # Step 1: Sample and apply parameters
            self._step_sample_parameters()
            
            # Step 2: Build KG
            if not self._step_build_kg():
                return self.result
            
            # Step 3: Run retrieval
            if not self._step_run_retrieval():
                return self.result
            
            # Step 4: Run benchmark
            if not self._step_run_benchmark():
                return self.result
            
            # Step 5: Copy metrics and plot
            self._step_copy_metrics_and_plot()
            
            # Success!
            self.result.status = "success"
            self.result.kg_build_time = time.time() - start_time
            self.log.success(f"Epoch {self.epoch} completed successfully!")
            self.log.info(f"  Accuracy: {self.result.accuracy:.2%}" if self.result.accuracy else "  Accuracy: N/A")
            self.log.info(f"  UUID: {self.result.graph_uuid}")
            
        except Exception as e:
            self.result.status = "error"
            self.result.error_message = str(e)
            self.log.error(f"Epoch {self.epoch} failed with exception: {e}")
        
        # Save epoch summary
        self._save_summary()
        
        return self.result
    
    def _step_sample_parameters(self):
        """Sample parameters and create modified config."""
        self.log.info("Step 1: Sampling parameters...")
        
        # Sample random parameters
        params = sample_parameters()
        self.result.parameters = params
        
        self.log.debug(f"  Sampled parameters: {params}")
        self.log.info(f"  Sampled parameters:")
        for name, value in params.items():
            self.log.info(f"    {name}: {value}")
        
        # Load base config and apply parameters
        config = load_yaml(self.base_config_path)
        apply_parameters_to_config(config, params)
        
        # Ensure subgraph_extraction_injection is enabled for subgraph runs
        set_nested_value(config, ["llm_injector", "subgraph_extraction_injection"], True)
        
        # Save modified config
        save_yaml(config, self.config_path)
        self.log.debug(f"  Config saved with subgraph_extraction_injection enabled")
        self.log.info(f"  Saved config to: {self.config_path}")
    
    def _step_build_kg(self) -> bool:
        """Build the knowledge graph."""
        self.log.info("Step 2: Building knowledge graph...")
        
        if self.dry_run:
            self.log.info("  [DRY RUN] Skipping KG build")
            self.result.graph_uuid = f"dry-run-{self.epoch:03d}"
            return True
        
        cmd = [
            "python3", "-m", "src.cli.main", "kg",
            "--config", str(self.config_path),
            "--vlm-output", VLM_OUTPUT_PATH,
        ]
        
        returncode, stdout, stderr = run_command(
            cmd,
            self.log,
            cwd=self.project_root,
            log_file=self.kg_log_path,
            timeout=7200,  # 2 hour timeout for KG build
        )
        
        if returncode != 0:
            self.result.status = "kg_build_failed"
            self.result.error_message = f"KG build failed with code {returncode}"
            self.log.error(f"  KG build failed. See log: {self.kg_log_path}")
            return False
        
        # Extract UUID from output
        with open(self.kg_log_path, 'r') as f:
            log_content = f.read()
        
        uuid = extract_uuid_from_output(log_content)
        if not uuid:
            self.result.status = "uuid_extraction_failed"
            self.result.error_message = "Could not extract UUID from KG build log"
            self.log.error("  Could not extract UUID from KG build output")
            return False
        
        self.result.graph_uuid = uuid
        self.log.info(f"  KG built successfully with UUID: {uuid}")
        return True
    
    def _step_run_retrieval(self) -> bool:
        """Run batch retrieval."""
        self.log.info("Step 3: Running batch retrieval...")
        
        if self.dry_run:
            self.log.info("  [DRY RUN] Skipping retrieval")
            return True
        
        cmd = [
            "python3", "-m", "src.cli.main", "batch-retrieve",
            "--config", str(self.config_path),
            "--graph-uuid", self.result.graph_uuid,
            "--input", RETRIEVAL_INPUT_PATH,
            "--output", str(self.retrieval_output_path),
        ]
        
        retrieval_log = self.epoch_dir / "retrieval.log"
        returncode, stdout, stderr = run_command(
            cmd,
            self.log,
            cwd=self.project_root,
            log_file=retrieval_log,
            timeout=3600,
        )
        
        if returncode != 0:
            self.result.status = "retrieval_failed"
            self.result.error_message = f"Retrieval failed with code {returncode}"
            self.log.error(f"  Retrieval failed. See log: {retrieval_log}")
            return False
        
        self.log.info(f"  Retrieval completed. Output: {self.retrieval_output_path}")
        
        # Try to get retrieval time from metrics
        metrics_dir = self.project_root / "metrics"
        avg_time = extract_retrieval_time(metrics_dir, self.result.graph_uuid)
        if avg_time:
            self.result.avg_retrieval_time = avg_time
            self.log.info(f"  Average retrieval time: {avg_time:.2f}s")
        
        return True
    
    def _step_run_benchmark(self) -> bool:
        """Run benchmark evaluation."""
        self.log.info("Step 4: Running benchmark...")
        
        if self.dry_run:
            self.log.info("  [DRY RUN] Skipping benchmark")
            self.result.accuracy = 0.0
            return True
        
        cmd = [
            "python3", "-m", "src.cli.main", "benchmark",
            "--config", str(self.config_path),
            "--input", str(self.retrieval_output_path),
            "--output", str(self.benchmark_output_path),
        ]
        
        benchmark_log = self.epoch_dir / "benchmark.log"
        returncode, stdout, stderr = run_command(
            cmd,
            self.log,
            cwd=self.project_root,
            log_file=benchmark_log,
            timeout=1800,
        )
        
        if returncode != 0:
            self.result.status = "benchmark_failed"
            self.result.error_message = f"Benchmark failed with code {returncode}"
            self.log.error(f"  Benchmark failed. See log: {benchmark_log}")
            return False
        
        # Extract accuracy
        accuracy, total, correct = extract_accuracy_from_benchmark(self.benchmark_output_path)
        if accuracy is not None:
            self.result.accuracy = accuracy
            self.result.total_queries = total
            self.result.correct_queries = correct
            self.log.info(f"  Benchmark accuracy: {accuracy:.2%} ({correct}/{total})")
        else:
            self.log.warning("  Could not extract accuracy from benchmark results")
        
        return True
    
    def _step_copy_metrics_and_plot(self):
        """Copy KG metrics and generate comparison plot."""
        self.log.info("Step 5: Copying metrics and generating plot...")
        
        if self.dry_run:
            self.log.info("  [DRY RUN] Skipping metrics copy and plot")
            return
        
        # Find and copy KG batch metrics
        metrics_dir = self.project_root / "metrics"
        kg_metrics_file = metrics_dir / f"{self.result.graph_uuid}_batch_metrics_kg.json"
        
        if kg_metrics_file.exists():
            dest_metrics = self.epoch_dir / f"kg_metrics_{self.result.graph_uuid}.json"
            shutil.copy(kg_metrics_file, dest_metrics)
            self.log.info(f"  Copied KG metrics to: {dest_metrics}")
            
            # Generate plot config
            plot_config = generate_plot_config(
                mvp_metrics_path=MVP_METRICS_PATH,
                current_metrics_path=str(dest_metrics),
                output_path=str(self.metrics_plot_path),
                epoch=self.epoch,
            )
            save_yaml(plot_config, self.plot_config_path)
            
            # Run plot script
            cmd = [
                "python3", "scripts/plot/plot_metrics.py",
                "--config", str(self.plot_config_path),
            ]
            
            returncode, _, _ = run_command(cmd, self.log, cwd=self.project_root)
            if returncode == 0:
                self.log.info(f"  Generated plot: {self.metrics_plot_path}")
            else:
                self.log.warning("  Plot generation failed (non-critical)")
        else:
            self.log.warning(f"  KG metrics file not found: {kg_metrics_file}")
        
        # Copy retrieval timing metrics if available
        for f in metrics_dir.glob(f"retrieval_times_{self.result.graph_uuid}_*.json"):
            dest = self.epoch_dir / f.name
            shutil.copy(f, dest)
            self.log.info(f"  Copied retrieval metrics: {dest}")
    
    def _save_summary(self):
        """Save epoch summary to JSON."""
        summary = asdict(self.result)
        summary['timestamp'] = datetime.now().isoformat()
        
        with open(self.epoch_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log.info(f"  Saved epoch summary: {self.epoch_summary_path}")


# ============================================================================
# MAIN SWEEP RUNNER
# ============================================================================

class SweepRunner:
    """Runs the complete parameter sweep across multiple epochs."""
    
    def __init__(
        self,
        num_epochs: int,
        base_config_path: Path,
        output_dir: Path,
        project_root: Path,
        dry_run: bool = False,
        start_epoch: int = 1,
    ):
        self.num_epochs = num_epochs
        self.base_config_path = base_config_path
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
        
        self.results: List[EpochResult] = []
    
    def run(self):
        """Run all epochs."""
        self.log.separator("=")
        self.log.info("PARAMETER SWEEP STARTING")
        self.log.info(f"  Epochs: {self.num_epochs}")
        self.log.info(f"  Output directory: {self.sweep_dir}")
        self.log.info(f"  Dry run: {self.dry_run}")
        self.log.separator("=")
        
        self.log.info("Parameter ranges:")
        for param in SWEEP_PARAMETERS:
            self.log.info(f"  {param.name}: [{param.min_val}, {param.max_val}]")
        
        sweep_start = time.time()
        
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            runner = EpochRunner(
                epoch=epoch,
                base_config_path=self.base_config_path,
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
        
        # Final summary
        self._print_final_summary(sweep_duration)
    
    def _save_sweep_summary(self):
        """Save sweep summary to JSON."""
        summary = {
            'num_epochs': self.num_epochs,
            'completed_epochs': len(self.results),
            'successful_epochs': sum(1 for r in self.results if r.status == "success"),
            'failed_epochs': sum(1 for r in self.results if r.status != "success"),
            'results': [asdict(r) for r in self.results],
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add best result
        successful = [r for r in self.results if r.accuracy is not None]
        if successful:
            best = max(successful, key=lambda r: r.accuracy)
            summary['best_result'] = {
                'epoch': best.epoch,
                'accuracy': best.accuracy,
                'parameters': best.parameters,
                'graph_uuid': best.graph_uuid,
            }
        
        summary_path = self.sweep_dir / "sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_final_summary(self, duration: float):
        """Print final sweep summary."""
        self.log.separator("=")
        self.log.info("PARAMETER SWEEP COMPLETED")
        self.log.separator("=")
        
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status != "success"]
        
        self.log.info(f"Total epochs: {len(self.results)}")
        self.log.info(f"Successful: {len(successful)}")
        self.log.info(f"Failed: {len(failed)}")
        self.log.info(f"Total duration: {duration/60:.1f} minutes")
        
        if successful:
            accuracies = [r.accuracy for r in successful if r.accuracy is not None]
            if accuracies:
                self.log.info(f"Accuracy range: {min(accuracies):.2%} - {max(accuracies):.2%}")
                self.log.info(f"Mean accuracy: {sum(accuracies)/len(accuracies):.2%}")
                
                best = max(successful, key=lambda r: r.accuracy or 0)
                self.log.info(f"Best result:")
                self.log.info(f"  Epoch: {best.epoch}")
                self.log.info(f"  Accuracy: {best.accuracy:.2%}")
                self.log.info(f"  UUID: {best.graph_uuid}")
                self.log.info(f"  Parameters: {best.parameters}")
        
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
        description="Run parameter sweep for KG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 15 epochs with default settings
  python3 scripts/run_parameter_sweep.py --epochs 15

  # Dry run to test the setup
  python3 scripts/run_parameter_sweep.py --epochs 3 --dry-run

  # Custom config and output directory
  python3 scripts/run_parameter_sweep.py --epochs 10 --config my_config.yaml --output-dir results/
        """
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to run (default: 15)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base_config.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sweeps",
        help="Base directory for sweep outputs (default: outputs/sweeps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing actual pipeline commands",
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
    
    # Run sweep
    runner = SweepRunner(
        num_epochs=args.epochs,
        base_config_path=base_config_path,
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
