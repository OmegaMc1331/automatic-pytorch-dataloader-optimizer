from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from apdo.errors import MemoryLimitExceeded


@dataclass(frozen=True, slots=True)
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int
    capability: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "capability": list(self.capability),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    processor: str
    cpu_count: int
    cpu_affinity_count: int | None
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    cuda_devices: tuple[GpuInfo, ...]
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    torch_num_threads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "cpu_affinity_count": self.cpu_affinity_count,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cuda_devices": [device.to_dict() for device in self.cuda_devices],
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "torch_num_threads": self.torch_num_threads,
        }


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    baseline_available_bytes: int | None
    peak_process_rss_bytes: int | None
    peak_system_used_bytes: int | None
    limit_bytes: int | None
    limit_exceeded: bool

    @property
    def limit_enforced(self) -> bool:
        return self.limit_bytes is not None and self.baseline_available_bytes is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_available_bytes": self.baseline_available_bytes,
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
            "peak_system_used_bytes": self.peak_system_used_bytes,
            "limit_bytes": self.limit_bytes,
            "limit_enforced": self.limit_enforced,
            "limit_exceeded": self.limit_exceeded,
        }


@cache
def _windows_memory_api() -> tuple[type[ctypes.Structure], Any]:
    import ctypes.wintypes as wt

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD),
            ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = wt.BOOL
    return MemoryStatus, kernel32


@cache
def _windows_process_api() -> tuple[type[ctypes.Structure], Any, Any]:
    import ctypes.wintypes as wt

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wt.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wt.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    return ProcessMemoryCounters, kernel32, psapi


def system_memory_bytes() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        memory_status, kernel32 = _windows_memory_api()
        status = memory_status()
        status.dwLength = ctypes.sizeof(status)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return None, None

    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}
        try:
            with open("/proc/meminfo", encoding="ascii") as meminfo:
                for line in meminfo:
                    key, _, remainder = line.partition(":")
                    if key in {"MemTotal", "MemAvailable"}:
                        values[key] = int(remainder.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return None, None
        return values.get("MemTotal"), values.get("MemAvailable")

    if sys.platform == "darwin":
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            total = None
        return total, None

    return None, None


def process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        counters_type, kernel32, psapi = _windows_process_api()
        counters = counters_type()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None

    return None


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        value = process_cpu_count()
        if value:
            return int(value)

    return max(1, os.cpu_count() or 1)


def collect_environment() -> EnvironmentInfo:
    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = None

    cuda_available = torch.cuda.is_available()
    cuda_devices: list[GpuInfo] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(index)
            except Exception:
                continue
            cuda_devices.append(
                GpuInfo(
                    index=index,
                    name=properties.name,
                    total_memory_bytes=int(properties.total_memory),
                    capability=(int(properties.major), int(properties.minor)),
                )
            )

    total_memory, available_memory = system_memory_bytes()
    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 1,
        cpu_affinity_count=affinity_count,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        cuda_devices=tuple(cuda_devices),
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        torch_num_threads=torch.get_num_threads(),
    )


class MemoryMonitor:
    def __init__(self, limit_bytes: int | None = None, interval_s: float = 0.05) -> None:
        self._limit_bytes = limit_bytes
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        _, self._baseline_available = system_memory_bytes()
        self._peak_process_rss = process_rss_bytes()
        self._peak_system_used = 0
        self._limit_exceeded = False
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="apdo-memory-monitor",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("MemoryMonitor.start() was called more than once")
        self._started = True
        self._sample_once()
        self._thread.start()

    def _sample_once(self) -> None:
        rss = process_rss_bytes()
        _, available = system_memory_bytes()
        with self._lock:
            if rss is not None:
                self._peak_process_rss = max(self._peak_process_rss or 0, rss)
            if available is not None and self._baseline_available is not None:
                used = max(0, self._baseline_available - available)
                self._peak_system_used = max(self._peak_system_used, used)
                if self._limit_bytes is not None and used > self._limit_bytes:
                    self._limit_exceeded = True

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._sample_once()

    def raise_if_exceeded(self) -> None:
        with self._lock:
            exceeded = self._limit_exceeded
            observed = self._peak_system_used
        if exceeded:
            limit_gb = (self._limit_bytes or 0) / 1_000_000_000
            observed_gb = observed / 1_000_000_000
            raise MemoryLimitExceeded(
                f"observed memory increase of {observed_gb:.2f} GB exceeds "
                f"max_ram_gb={limit_gb:.2f}"
            )

    def stop(self) -> MemorySnapshot:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._sample_once()
        with self._lock:
            peak_system_used = (
                self._peak_system_used if self._baseline_available is not None else None
            )
            return MemorySnapshot(
                baseline_available_bytes=self._baseline_available,
                peak_process_rss_bytes=self._peak_process_rss,
                peak_system_used_bytes=peak_system_used,
                limit_bytes=self._limit_bytes,
                limit_exceeded=self._limit_exceeded,
            )
