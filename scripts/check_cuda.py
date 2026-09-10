from __future__ import annotations

import platform
import sys

import torch


def gibibytes(value: int) -> float:
    return value / (1024 ** 3)


def main() -> None:
    print(f"os={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"cuda_available={available}")
    print(f"gpu_count={torch.cuda.device_count() if available else 0}")
    if not available:
        raise SystemExit("CUDA is unavailable; stop before GPU smoke")

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info()
        print(f"gpu_{index}_name={properties.name}")
        print(f"gpu_{index}_total_gib={gibibytes(total):.2f}")
        print(f"gpu_{index}_free_gib={gibibytes(free):.2f}")


if __name__ == "__main__":
    main()
