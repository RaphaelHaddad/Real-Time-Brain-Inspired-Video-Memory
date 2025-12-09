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


@dataclass
class FloatParameterRange:
    """Defines the range for a float parameter to sweep."""
    name: str
    min_val: float
    max_val: float
    yaml_path: List[str]


# Parameters to sweep with their ranges
SWEEP_PARAMETERS = [
    ParameterRange("max_connection_subgraph", 2, 3, ["chunking", "max_connection_subgraph"]),
    ParameterRange("max_new_triplets", 3, 25, ["chunking", "max_new_triplets"]),  # Reduced from 25
    ParameterRange("max_inter_chunk_relations", 1, 20, ["chunking", "max_inter_chunk_relations"]),  # Reduced from 20
    ParameterRange("max_merge_instructions", 1, 15, ["chunking", "max_merge_instructions"]),  # Reduced from 12
    ParameterRange("max_prune_instructions", 1, 20, ["chunking", "max_prune_instructions"]),  # Reduced from 20
    # Retrieval sweep parameters
    ParameterRange("top_k", 4, 12, ["retrieval", "top_k"]),
    ParameterRange("graph_hops", 2, 5, ["retrieval", "graph_hops"]),
]

# Community-specific sweep parameter ranges. These will be applied only
# when community_high_graph.community_creator and community_high_graph.community_retriever
# are enabled in the base config. question_per_chunk is an integer in [1..6].
# community_resolution is the Leiden resolution/gamma parameter. We use the range [0.1, 2.0] to capture coarse-to-fine
# partitions. Higher gamma leads to a higher resolution (more smaller communities), smaller gamma leads to fewer larger communities.
COMMUNITY_SWEEP_INT_PARAMETERS = [
    ParameterRange("question_per_chunk", 1, 6, ["community_high_graph", "question_per_chunk"]),
]

COMMUNITY_SWEEP_FLOAT_PARAMETERS = [
    FloatParameterRange("community_resolution", 0.1, 2.0, ["community_high_graph", "community_resolution"]),
]

# Global float parameters to sweep (retrieval-level floats)
GLOBAL_SWEEP_FLOAT_PARAMETERS = [
    # compression_threshold: cosine similarity cutoff used by post-compression.
    # Range chosen to explore from effectively disabled compression (0.0)
    # to moderately aggressive filtering (0.4). Default in base_config is 0.15.
    FloatParameterRange("compression_threshold", 0.0, 0.4, ["retrieval", "compression_threshold"]),
]

# Fixed paths
MVP_METRICS_PATH = "data/metrics/mvp_93e9c82e-95d6-4864-8ac1-2ae70edfd961.json"
PLOT_CONFIG_TEMPLATE = "config/plot_metrics.yaml"

# Hop method choices for all-retrieval mode
HOP_METHODS = ["naive", "page_rank", "ch3_l3"]


@dataclass
class RetrievalConfig:
    """Configuration for a single retrieval run."""
    community_retriever: bool
    hop_method: str
    top_k: int = 9
    graph_hops: int = 2
    top_k_hop_pagerank: int = 10
    top_k_hop_ch3_l3: int = 4
    max_path_length_ch3: int = 4
    cum_score_aggregation: str = "product"


@dataclass
class EpochResult:
    """Results from a single epoch."""
    epoch: int
    graph_uuid: str
    parameters: Dict[str, Any]
    accuracy: Optional[float] = None
    total_queries: int = 0
    correct_queries: int = 0
    avg_retrieval_time: Optional[float] = None
    kg_build_time: Optional[float] = None
    status: str = "not_started"
    error_message: Optional[str] = None
    output_folder: str = ""
    retrieval_results: List[Dict[str, Any]] = field(default_factory=list)  # List of retrieval run results


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


def sample_float_parameters() -> Dict[str, float]:
    """Sample random values for float sweep parameters (community resolution).

    Returns a dict mapping parameter name to float value.
    """
    params = {}
    # Sample community float parameters
    for param in COMMUNITY_SWEEP_FLOAT_PARAMETERS:
        params[param.name] = round(random.uniform(param.min_val, param.max_val), 3)

    # Sample global float parameters as well
    for param in GLOBAL_SWEEP_FLOAT_PARAMETERS:
        params[param.name] = round(random.uniform(param.min_val, param.max_val), 3)

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
        vlm_output_path: Path,
        retrieval_input_path: Path,
        log: Logger,
        dry_run: bool = False,
        all_retrieval: bool = False,
        comparable_all_retrieval: bool = False,
        derive_hop_params: bool = False,
        expected_chunk_json: Optional[Path] = None,
    ):
        self.epoch = epoch
        self.base_config_path = base_config_path
        self.project_root = project_root
        self.vlm_output_path = vlm_output_path
        self.retrieval_input_path = retrieval_input_path
        self.log = log
        self.dry_run = dry_run
        self.all_retrieval = all_retrieval
        self.comparable_all_retrieval = comparable_all_retrieval
        self.derive_hop_params = derive_hop_params
        self._precomputed_methods: set[str] = set()
        self.expected_chunk_json = expected_chunk_json
        
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
            
            # Step 3: Run retrieval(s)
            if self.all_retrieval:
                if not self._step_run_all_retrievals():
                    return self.result
            else:
                if not self._step_run_retrieval():
                    return self.result
            
            # Step 4: Run benchmark (only if not all_retrieval)
            if not self.all_retrieval:
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
        
        # Load base config and apply parameters (core sweep params)
        config = load_yaml(self.base_config_path)
        apply_parameters_to_config(config, params)

        # Sample global float sweep parameters (e.g., compression_threshold)
        for fparam in GLOBAL_SWEEP_FLOAT_PARAMETERS:
            sampled_val = round(random.uniform(fparam.min_val, fparam.max_val), 3)
            set_nested_value(config, fparam.yaml_path, sampled_val)
            params[fparam.name] = sampled_val
            self.log.info(f"  {fparam.name}: {sampled_val} (sampled in range [{fparam.min_val}, {fparam.max_val}])")

    # If community_high_graph is enabled for both creator and retriever,
        # sample community-specific parameters and apply them.
        community_config = get_nested_value(config, ["community_high_graph"], {})
        if community_config and community_config.get("community_creator") and community_config.get("community_retriever"):
            self.log.info("Community creator and retriever enabled — sampling community parameters...")
            # Sample integer community params
            for cparam in COMMUNITY_SWEEP_INT_PARAMETERS:
                sampled = random.randint(cparam.min_val, cparam.max_val)
                set_nested_value(config, cparam.yaml_path, sampled)
                params[cparam.name] = sampled

            # Sample float community params
            float_params = sample_float_parameters()
            for name, value in float_params.items():
                # Find corresponding yaml path and write
                for fparam in COMMUNITY_SWEEP_FLOAT_PARAMETERS:
                    if fparam.name == name:
                        set_nested_value(config, fparam.yaml_path, value)
                        # Save rounded value for result metadata
                        params[name] = value
                        # Explain impact of gamma to the user via logs
                        self.log.info(
                            f"  community_resolution (gamma) sampled: {value} - "
                            "Higher gamma -> higher resolution -> more smaller communities; "
                            "Lower gamma -> lower resolution -> fewer larger communities"
                        )
                        break
        
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
            "--vlm-output", str(self.vlm_output_path),
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
    
    def _step_run_all_retrievals(self) -> bool:
        """Run multiple retrieval configurations (all hop methods, with/without community)."""
        self.log.info("Step 3: Running all retrieval configurations...")
        self.log.info(f"  Total configurations: 2 (community on/off) × 3 (hop methods) = 6 runs")
        
        if self.dry_run:
            self.log.info("  [DRY RUN] Skipping all retrievals")
            return True
        
        config_run = 0
        # Read baseline retrieval params from the saved config for comparability
        base_config = load_yaml(self.config_path)
        base_top_k = get_nested_value(base_config, ["retrieval", "top_k"], 9)
        base_graph_hops = get_nested_value(base_config, ["retrieval", "graph_hops"], 2)
        self.log.info(f"  Baseline top_k={base_top_k}, graph_hops={base_graph_hops} (shared across runs)")
        # First: community_retriever = False, then True
        for community_enabled in [False, True]:
            community_label = "community_on" if community_enabled else "community_off"
            self.log.info(f"\n  [{community_label.upper()}] Running retrievals with community_retriever={community_enabled}")
            
            for hop_method in HOP_METHODS:
                config_run += 1
                self.log.separator("-")
                self.log.info(f"  Run {config_run}/6: hop_method={hop_method}, {community_label}")
                
                # Determine retrieval parameters
                if self.comparable_all_retrieval:
                    retrieval_params = self._comparable_retrieval_params(
                        hop_method, base_top_k, base_graph_hops
                    )
                else:
                    retrieval_params = self._sample_retrieval_params(hop_method)
                retrieval_config = RetrievalConfig(
                    community_retriever=community_enabled,
                    hop_method=hop_method,
                    **retrieval_params
                )
                
                # Create run-specific output path
                run_output_path = self.epoch_dir / f"retrieval_results_{community_label}_{hop_method}.json"
                run_benchmark_path = self.epoch_dir / f"benchmark_results_{community_label}_{hop_method}.json"
                
                # Create modified config for this run
                run_config_path = self.epoch_dir / f"config_{community_label}_{hop_method}.yaml"
                self._write_retrieval_config(run_config_path, retrieval_config)
                
                # Run retrieval
                if not self._run_single_retrieval(run_config_path, run_output_path):
                    self.log.warning(f"  Retrieval failed for {community_label}_{hop_method}")
                    continue
                
                # Run benchmark for this retrieval
                accuracy, total, correct = self._run_single_benchmark(run_config_path, run_output_path, run_benchmark_path)
                
                # Store result
                result_entry = {
                    "run": config_run,
                    "community_retriever": community_enabled,
                    "hop_method": hop_method,
                    "parameters": asdict(retrieval_config),
                    "accuracy": accuracy,
                    "total_queries": total,
                    "correct_queries": correct,
                    "output": str(run_output_path),
                    "benchmark": str(run_benchmark_path),
                }
                self.result.retrieval_results.append(result_entry)
                
                if accuracy is not None:
                    self.log.info(f"    Accuracy: {accuracy:.2%} ({correct}/{total})")
                else:
                    self.log.warning(f"    Could not extract accuracy")
        
        return True
    
    def _comparable_retrieval_params(self, hop_method: str, base_top_k: int, base_graph_hops: int) -> Dict[str, Any]:
        """Return retrieval params derived from shared baseline values for comparability."""
        params: Dict[str, Any] = {}
        if hop_method == "naive":
            params["top_k"] = base_top_k
            params["graph_hops"] = base_graph_hops
        elif hop_method == "page_rank":
            # Derive pagerank hop top_k from baseline when enabled
            pr_top_k = base_top_k * 2 if self.derive_hop_params else max(4, base_top_k)
            params["top_k_hop_pagerank"] = pr_top_k
        elif hop_method == "ch3_l3":
            ch_top_k = base_top_k if self.derive_hop_params else max(2, base_top_k)
            max_path_len = base_graph_hops if self.derive_hop_params else max(2, base_graph_hops)
            params["top_k_hop_ch3_l3"] = ch_top_k
            params["max_path_length_ch3"] = max_path_len
            params["cum_score_aggregation"] = "product"
        return params

    def _sample_retrieval_params(self, hop_method: str) -> Dict[str, Any]:
        """Sample retrieval parameters based on hop method."""
        params = {}
        if hop_method == "naive":
            params["top_k"] = random.randint(5, 30)
            params["graph_hops"] = random.randint(2, 5)
        elif hop_method == "page_rank":
            params["top_k_hop_pagerank"] = random.randint(4, 20)
        elif hop_method == "ch3_l3":
            params["top_k_hop_ch3_l3"] = random.randint(2, 8)
            params["max_path_length_ch3"] = random.randint(2, 8)
            params["cum_score_aggregation"] = random.choice(["sum", "product"])
        return params
    
    def _write_retrieval_config(self, config_path: Path, retrieval_config: RetrievalConfig):
        """Write a config file with the given retrieval configuration."""
        config = load_yaml(self.config_path)
        
        # Update retrieval settings
        set_nested_value(config, ["retrieval", "hop_method"], retrieval_config.hop_method)
        set_nested_value(config, ["community_high_graph", "community_retriever"], retrieval_config.community_retriever)
        
        # Set hop-method-specific parameters
        if retrieval_config.hop_method == "naive":
            set_nested_value(config, ["retrieval", "top_k"], retrieval_config.top_k)
            set_nested_value(config, ["retrieval", "graph_hops"], retrieval_config.graph_hops)
        elif retrieval_config.hop_method == "page_rank":
            set_nested_value(config, ["retrieval", "top_k_hop_pagerank"], retrieval_config.top_k_hop_pagerank)
        elif retrieval_config.hop_method == "ch3_l3":
            set_nested_value(config, ["retrieval", "top_k_hop_ch3_l3"], retrieval_config.top_k_hop_ch3_l3)
            set_nested_value(config, ["retrieval", "max_path_length_ch3"], retrieval_config.max_path_length_ch3)
            set_nested_value(config, ["retrieval", "cum_score_aggregation"], retrieval_config.cum_score_aggregation)
        
        save_yaml(config, config_path)
    
    def _run_single_retrieval(self, config_path: Path, output_path: Path) -> bool:
        """Run a single retrieval with the given config."""
        
        # Load config to check hop method
        config = load_yaml(config_path)
        hop_method = get_nested_value(config, ["retrieval", "hop_method"])
        
        # Run precomputation if needed
        if hop_method in ["page_rank", "ch3_l3"] and hop_method not in self._precomputed_methods:
            self.log.info(f"  Running precomputation for {hop_method}...")
            precompute_cmd = [
                "python3", "-m", "src.cli.main", "precompute",
                "--config", str(config_path),
                "--graph-uuid", self.result.graph_uuid,
                "--methods", hop_method,
            ]
            
            precompute_log = self.epoch_dir / f"precompute_{hop_method}.log"
            returncode, stdout, stderr = run_command(
                precompute_cmd,
                self.log,
                cwd=self.project_root,
                log_file=precompute_log,
                timeout=1800,  # 30 minutes timeout for precomputation
            )
            
            if returncode != 0:
                self.log.error(f"  Precomputation failed for {hop_method}. See log: {precompute_log}")
                return False
            
            self.log.info(f"  Precomputation completed for {hop_method}")
            self._precomputed_methods.add(hop_method)
        
        # Run retrieval
        cmd = [
            "python3", "-m", "src.cli.main", "batch-retrieve",
            "--config", str(config_path),
            "--graph-uuid", self.result.graph_uuid,
            "--input", str(self.retrieval_input_path),
            "--output", str(output_path),
        ]
        # If expected chunk JSON is provided, include it to enable deep analysis
        if self.expected_chunk_json:
            cmd.extend(["--expected-chunk-json", str(self.expected_chunk_json)])
            # Also instruct CLI to place deep analysis under this epoch's deep_retrieval folder
            deep_dir = self.epoch_dir / "deep_retrieval"
            cmd.extend(["--deep-analysis-dir", str(deep_dir)])
            # Copy expected JSON into epoch folder for traceability
            try:
                dst = self.epoch_dir / ("expected_chunks_" + Path(self.expected_chunk_json).name)
                if not dst.exists():
                    import shutil

                    shutil.copy(self.expected_chunk_json, dst)
                    self.log.info(f"  Copied expected-chunk-json to epoch folder: {dst}")
            except Exception as e:
                self.log.warning(f"  Could not copy expected-chunk-json to epoch dir: {e}")
        
        retrieval_log = self.epoch_dir / f"retrieval_{output_path.stem}.log"
        returncode, stdout, stderr = run_command(
            cmd,
            self.log,
            cwd=self.project_root,
            log_file=retrieval_log,
            timeout=3600,
        )
        
        if returncode != 0:
            self.log.error(f"  Retrieval failed. See log: {retrieval_log}")
            return False
        
        return True
    
    def _run_single_benchmark(self, config_path: Path, retrieval_output: Path, benchmark_output: Path) -> Tuple[Optional[float], int, int]:
        """Run benchmark for a single retrieval output."""
        cmd = [
            "python3", "-m", "src.cli.main", "benchmark",
            "--config", str(config_path),
            "--input", str(retrieval_output),
            "--output", str(benchmark_output),
        ]
        
        benchmark_log = self.epoch_dir / f"benchmark_{benchmark_output.stem}.log"
        returncode, stdout, stderr = run_command(
            cmd,
            self.log,
            cwd=self.project_root,
            log_file=benchmark_log,
            timeout=1800,
        )
        
        if returncode != 0:
            self.log.error(f"  Benchmark failed. See log: {benchmark_log}")
            return None, 0, 0
        
        # Extract accuracy
        accuracy, total, correct = extract_accuracy_from_benchmark(benchmark_output)
        return accuracy, total, correct
    
    def _step_run_retrieval(self) -> bool:
        """Run batch retrieval (standard mode)."""
        self.log.info("Step 3: Running batch retrieval...")
        
        if self.dry_run:
            self.log.info("  [DRY RUN] Skipping retrieval")
            return True
        
        cmd = [
            "python3", "-m", "src.cli.main", "batch-retrieve",
            "--config", str(self.config_path),
            "--graph-uuid", self.result.graph_uuid,
            "--input", str(self.retrieval_input_path),
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
        vlm_output_path: Path,
        retrieval_input_path: Path,
        dry_run: bool = False,
        start_epoch: int = 1,
        all_retrieval: bool = False,
        comparable_all_retrieval: bool = False,
        derive_hop_params: bool = False,
        expected_chunk_json: Optional[Path] = None,
    ):
        self.num_epochs = num_epochs
        self.base_config_path = base_config_path
        self.output_dir = output_dir
        self.project_root = project_root
        self.vlm_output_path = vlm_output_path
        self.retrieval_input_path = retrieval_input_path
        self.dry_run = dry_run
        self.start_epoch = start_epoch
        self.all_retrieval = all_retrieval
        self.comparable_all_retrieval = comparable_all_retrieval
        self.derive_hop_params = derive_hop_params
        self.expected_chunk_json = expected_chunk_json
        
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
                vlm_output_path=self.vlm_output_path,
                retrieval_input_path=self.retrieval_input_path,
                log=self.log,
                dry_run=self.dry_run,
                all_retrieval=self.all_retrieval,
                comparable_all_retrieval=self.comparable_all_retrieval,
                derive_hop_params=self.derive_hop_params,
                expected_chunk_json=self.expected_chunk_json,
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
    parser.add_argument(
        "--vlm-output",
        type=str,
        default="data/outputs/vlm_output.json",
        help="Path to VLM output JSON file (default: data/outputs/vlm_output.json)",
    )
    parser.add_argument(
        "--retrieval-input",
        type=str,
        default="data/groundtruth/retrieval_offline.json",
        help="Path to retrieval groundtruth JSON file (default: data/groundtruth/retrieval_offline.json)",
    )
    parser.add_argument(
        "--expected-chunk-json",
        type=str,
        default=None,
        help="(Optional) JSON file with expected chunk ids per query for deep analysis",
    )
    parser.add_argument(
        "--all-retrieval",
        action="store_true",
        help="Run all retrieval configurations: 2 community modes × 3 hop methods = 6 retrievals per KG build",
    )
    parser.add_argument(
        "--comparable-all-retrieval",
        action="store_true",
        help="Share baseline retrieval params across all six runs per epoch; vary only hop-specific knobs",
    )
    parser.add_argument(
        "--derive-hop-params",
        action="store_true",
        help="Enable optional derivations: pagerank top_k ≈ 2×top_k_entities; ch3_l3 top_k ≈ top_k_entities; max_path_length_ch3 ≈ graph_hops",
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
    retrieval_input = project_root / args.retrieval_input
    if not retrieval_input.exists():
        print(f"Error: Retrieval input file not found: {retrieval_input}")
        sys.exit(1)

    expected_chunk_json = None
    if args.expected_chunk_json:
        expected_chunk_json = project_root / args.expected_chunk_json
        if not expected_chunk_json.exists():
            print(f"Error: expected-chunk-json file not found: {expected_chunk_json}")
            sys.exit(1)
    
    # Run sweep
    runner = SweepRunner(
        num_epochs=args.epochs,
        base_config_path=base_config_path,
        output_dir=output_dir,
        project_root=project_root,
        vlm_output_path=Path(args.vlm_output),
        retrieval_input_path=retrieval_input,
        dry_run=args.dry_run,
        start_epoch=args.start_epoch,
        all_retrieval=args.all_retrieval,
        comparable_all_retrieval=args.comparable_all_retrieval,
        derive_hop_params=args.derive_hop_params,
        expected_chunk_json=expected_chunk_json,
    )
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user. Partial results saved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
