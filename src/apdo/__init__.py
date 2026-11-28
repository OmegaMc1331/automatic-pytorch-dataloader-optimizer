from apdo._version import __version__
from apdo.benchmark import ConfigMeasurement, CycleMetrics, RoundMeasurement
from apdo.budget import BudgetController
from apdo.cache import CacheEntry, ConfigCache
from apdo.comparability import BatchObservations, ComparabilityResult, compare_measurements
from apdo.config import (
    BENCHMARK_PROTOCOL_VERSION,
    CACHE_POLICIES,
    COMPARABILITY_POLICIES,
    SEARCH_STRATEGIES,
    SELECTION_METRICS,
    TIMING_PROFILES,
    BenchmarkSettings,
    LoaderConfig,
)
from apdo.errors import (
    ApdoError,
    BenchmarkError,
    BudgetExhausted,
    ConfigurationError,
    DatasetError,
    FingerprintError,
    MemoryLimitExceeded,
)
from apdo.isolation import ReproducibilityInfo
from apdo.loader import LoaderFactory, LoaderSource, LoaderTemplate
from apdo.metrics import MetricsSummary
from apdo.optimizer import optimize
from apdo.order import RoundOrder
from apdo.profitability import ProfitabilityEstimate
from apdo.report import OptimizationResult
from apdo.resources import EnvironmentInfo, GpuInfo
from apdo.search import SearchTrace
from apdo.stability import StabilityReport, StabilityRun, StabilitySummary, evaluate_stability
from apdo.training import TrainingStepFactory
from apdo.validation import (
    VALIDATION_STATUSES,
    ValidationObservation,
    ValidationResult,
)

__all__ = [
    "BENCHMARK_PROTOCOL_VERSION",
    "CACHE_POLICIES",
    "SEARCH_STRATEGIES",
    "VALIDATION_STATUSES",
    "ApdoError",
    "BenchmarkError",
    "BenchmarkSettings",
    "BatchObservations",
    "BudgetController",
    "BudgetExhausted",
    "COMPARABILITY_POLICIES",
    "CacheEntry",
    "ComparabilityResult",
    "ConfigCache",
    "ConfigMeasurement",
    "ConfigurationError",
    "CycleMetrics",
    "DatasetError",
    "FingerprintError",
    "EnvironmentInfo",
    "GpuInfo",
    "LoaderConfig",
    "LoaderFactory",
    "LoaderSource",
    "LoaderTemplate",
    "MemoryLimitExceeded",
    "MetricsSummary",
    "OptimizationResult",
    "ProfitabilityEstimate",
    "RoundMeasurement",
    "ReproducibilityInfo",
    "RoundOrder",
    "SELECTION_METRICS",
    "SearchTrace",
    "TIMING_PROFILES",
    "TrainingStepFactory",
    "ValidationObservation",
    "ValidationResult",
    "StabilityReport",
    "StabilityRun",
    "StabilitySummary",
    "__version__",
    "compare_measurements",
    "evaluate_stability",
    "optimize",
]
