#!/usr/bin/env python3
"""Print a compact Python, PyTorch, and CUDA environment report."""

from __future__ import annotations

import platform
import sys


def gibibytes(byte_count: int) -> float:
    """Convert bytes to GiB."""

    return byte_count / (1024**3)


def main() -> int:
    print(f"Python version: {platform.python_version()} ({sys.executable})")

    try:
        import torch
    except (ImportError, OSError) as exc:
        print("PyTorch version: not installed or failed to load")
        print("CUDA available: False")
        print("GPU name: unavailable")
        print("VRAM: unavailable")
        print(f"PyTorch detail: {exc}")
        return 0

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("GPU name: unavailable")
        print("VRAM: unavailable")
        cuda_build = getattr(torch.version, "cuda", None)
        print(f"PyTorch CUDA build: {cuda_build or 'CPU-only/unknown'}")
        return 0

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index} name: {properties.name}")
        print(f"GPU {index} VRAM: {gibibytes(properties.total_memory):.2f} GiB")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
