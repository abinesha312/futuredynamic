"""
FutureDynamic: Adaptive Memory Management for Efficient LLM Inference Across GPU Architectures
"""

from .core import FutureDynamic
from .predictor import PredictorModule
from .offloader import OffloaderModule
from .coordinator import CoordinatorModule
from .energy import EnergyMonitor
from .benchmark import benchmark_futuredynamic

__version__ = "0.1.0"
__all__ = [
    "FutureDynamic",
    "PredictorModule",
    "OffloaderModule",
    "CoordinatorModule", 
    "EnergyMonitor",
    "benchmark_futuredynamic"
]
