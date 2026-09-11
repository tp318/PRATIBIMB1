"""
AeroTwin-4 Phase 5 Anomaly Detection Package.
"""

from .features import FeatureExtractor
from .splits import RunSplitter
from .preprocessing import FeatureScaler
from .statistical import StatisticalAnomalyDetector
from .isolation_forest import IsolationForestAnomalyDetector
try:
    from .autoencoder import AutoencoderAnomalyDetector
except ImportError:
    AutoencoderAnomalyDetector = None
from .scoring import AnomalyScorer
from .evaluation import Evaluator

__all__ = [
    "FeatureExtractor",
    "RunSplitter",
    "FeatureScaler",
    "StatisticalAnomalyDetector",
    "IsolationForestAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "AnomalyScorer",
    "Evaluator",
]
