from utils import get_device, benchmark, check_equal
import torch
import triton
import triton.language as tl
from typing import Callable


def manual_relu(x):
    return torch.maximum(x, torch.zeros_like(x))


def run_operation_unary(dim: int, operation: Callable) -> Callable:
    """Helper to create a closure for benchmarking unary ops."""
    x = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x)

# --- TRITON KERNEL ---

# We let Triton try different configs to find the fastest one for your GPU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def triton_relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. Program ID indicates which "chunk" of the vector this instance processes
    pid = tl.program_id(axis=0)
    
    # 2. Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # 3. Create offsets for the specific elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 4. Create a mask to prevent out-of-bounds memory access 
    # (crucial if n_elements is not a multiple of BLOCK_SIZE)
    mask = offsets < n_elements

    # 5. Load, Compute, Store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor):
    assert x.is_cuda
    
    x = x.contiguous()
    
    n_elements = x.numel()
    y = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    triton_relu_kernel[grid](
        x, y,
        n_elements,
    )
    return y


# --- RUNNER ---

def benchmark_relu():
    # Use a larger dim to ensure we test memory boundaries
    # 4096 * 4096 = ~16 million elements
    dim = 4096 * 4 
    print(f"--- Benchmarking ReLU Variants (dim={dim}x{dim}) ---")

    # Verify correctness first
    print("Verifying correctness...")
    check_equal(triton_relu, manual_relu, dim=dim)

    # Run Benchmarks
    benchmark("1. Manual PyTorch ReLU", run_operation_unary(dim, manual_relu))
    benchmark("2. Triton Custom ReLU ", run_operation_unary(dim, triton_relu))

if __name__ == "__main__":
    benchmark_relu()
