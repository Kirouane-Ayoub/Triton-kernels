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


def matmul_check_equal(func1, func2, M, N, K):
    """Checks if two functions produce the same output for MatMul."""
    A = torch.randn(M, K, device=get_device())
    B = torch.randn(K, N, device=get_device())

    # Ensure MatMul input is contiguous for correct stride calculation
    A = A.contiguous()
    B = B.contiguous()
    
    try:
        Y1 = func1(A, B)
        Y2 = func2(A, B)
    except Exception as e:
        print(f"❌ Error during execution of {func1.__name__} or {func2.__name__}: {e}")
        return

    # MatMul tends to have higher deviation, so we use a slightly looser tolerance
    if torch.allclose(Y1, Y2, atol=1e-2, rtol=1e-2):
        print(f"✅ {func1.__name__} and {func2.__name__} match!")
    else:
        diff = (Y1 - Y2).abs().max()
        print(f"❌ {func1.__name__} and {func2.__name__} mismatch! Max diff: {diff}")

def run_operation_binary(M, N, K, operation: Callable) -> Callable:
    """Helper to create a closure for benchmarking MatMul."""
    A = torch.randn(M, K, device=get_device())
    B = torch.randn(K, N, device=get_device())
    # Ensure contiguous for fair benchmarking
    A = A.contiguous()
    B = B.contiguous()
    return lambda: operation(A, B)
