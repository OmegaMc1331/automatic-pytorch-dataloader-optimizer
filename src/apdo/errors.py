class ApdoError(Exception):
    pass


class ConfigurationError(ApdoError, ValueError):
    pass


class DatasetError(ApdoError, ValueError):
    pass


class BenchmarkError(ApdoError, RuntimeError):
    pass


class MemoryLimitExceeded(BenchmarkError):
    pass
