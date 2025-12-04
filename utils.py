import torch
import time
from typing import Callable


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_equal(func1, func2, dim=4096):
    """Checks if two functions produce the same output."""
    x = torch.randn(dim, dim, device=get_device())
    y1 = func1(x)
    y2 = func2(x)
    if torch.allclose(y1, y2, atol=1e-3, rtol=1e-3):
        print(f"✅ {func1.__name__} and {func2.__name__} match!")
    else:
        print(f"❌ {func1.__name__} and {func2.__name__} mismatch! Max diff: {(y1 - y2).abs().max()}")


def benchmark(description: str, run: Callable, num_warmups: int = 10, num_trials: int = 20):
    """Benchmarks a function by measuring wall-clock time."""
    if not torch.cuda.is_available():
        print("Skipping benchmark (No GPU)")
        return

    # Warmup
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(num_trials):
        start = time.time()
        run()
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000) # Convert to ms

    avg_time = sum(times) / len(times)
    print(f"{description}: {avg_time:.4f} ms")
    return avg_time


def run_benchmark(name, func, x):
    # Warmup
    for _ in range(10):
        func(x)
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(100):
        func(x)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"{name}: {(end - start) * 1000 / 100:.4f} ms")


def run_operation_unary(dim: int, operation: Callable) -> Callable:
    """Helper to create a closure for benchmarking unary ops (like GeLU)."""
    x = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x)