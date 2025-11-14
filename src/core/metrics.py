import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
from .logger import get_logger

logger = get_logger(__name__)

class MetricsTracker:
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "start_time": time.time(),
            "timings": {},
            "counts": {},
            "batch_metrics": []
        }

    def record_timing(self, category: str, operation: str, duration: float):
        """Record timing for a specific operation"""
        key = f"{category}.{operation}"
        if key not in self.metrics["timings"]:
            self.metrics["timings"][key] = []
        self.metrics["timings"][key].append(duration)
        logger.debug(f"Recorded timing: {key} = {duration:.4f}s")

    def record_count(self, category: str, operation: str, count: int):
        """Record a count for a specific operation"""
        key = f"{category}.{operation}"
        if key not in self.metrics["counts"]:
            self.metrics["counts"][key] = 0
        self.metrics["counts"][key] += count
        logger.debug(f"Recorded count: {key} = {count}")

    def add_batch_metrics(self, batch_metrics: Dict[str, Any]):
        """Add batch-specific metrics"""
        self.metrics["batch_metrics"].append(batch_metrics)
        logger.debug(f"Added batch metrics: {batch_metrics}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        summary = {
            "total_runtime": time.time() - self.metrics["start_time"],
            "timing_averages": {}
        }
        
        # Calculate average timings
        for key, values in self.metrics["timings"].items():
            if values:
                summary["timing_averages"][key] = sum(values) / len(values)
        
        summary["counts"] = self.metrics["counts"]
        summary["batch_count"] = len(self.metrics["batch_metrics"])
        
        return summary

    def save_metrics(self, path: str):
        """Save metrics to a JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics["end_time"] = time.time()
        self.metrics["summary"] = self.get_summary()
        
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to: {path}")