import platform
import torch
import os
from pathlib import Path

class PlatformUtils:
    @staticmethod
    def get_optimal_workers() -> int:
        """Get optimal worker count based on CPU cores"""
        cpu_count = os.cpu_count() or 4
        return min(cpu_count, 8)

    @staticmethod
    def get_video_backend() -> str:
        """Determine best video processing backend"""
        system = platform.system().lower()
        if system == "windows" and "WSL" in platform.release():
            return "ffmpeg"  # Better WSL support
        elif system == "darwin":  # Mac with Apple Silicon
            return "avfoundation"
        return "opencv"  # Linux default

    @staticmethod
    def get_device() -> str:
        """Detect best available hardware acceleration"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
        return "cpu"

    @staticmethod
    def get_data_dir() -> Path:
        """Get platform-appropriate data directory"""
        if platform.system() == "Windows":
            return Path(os.environ.get("APPDATA", "~\\AppData\\Roaming")) / "vidgraph"
        elif platform.system() == "Darwin":
            return Path("~/Library/Application Support/vidgraph").expanduser()
        else:  # Linux
            return Path("~/.local/share/vidgraph").expanduser()