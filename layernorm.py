import torch
import triton
import triton.language as tl
from utils import get_device, benchmark, check_equal, run_operation_unary


# 1. Manual LayerNorm
def manual_layernorm(x):
    # Memory bound: multiple reads/writes to HBM
    eps = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)
    return (x - mean) / torch.sqrt(var + eps)


# 2. PyTorch Native
def pytorch_layernorm(x):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],))


# 3. Triton LayerNorm
# Key idea: each program handles one row.
# Single pass using Welford's online algorithm to compute mean and variance
# simultaneously, then normalize in a second pass.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['num_cols'],
)
@triton.jit
def triton_layernorm_kernel(
    x_ptr, y_ptr,
    x_row_stride, y_row_stride,
    num_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_x = x_ptr + row_idx * x_row_stride
    row_start_y = y_ptr + row_idx * y_row_stride
    offs = tl.arange(0, BLOCK_SIZE)

    # --- PASS 1: accumulate sum and sum-of-squares ---
    # Simpler and faster than Welford: just two running accumulators,
    # derive mean and variance once at the end.
    # var(x) = E[x^2] - E[x]^2
    _sum = 0.0
    _sum_sq = 0.0

    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols
        x_chunk = tl.load(row_start_x + col_idxs, mask=mask, other=0.0)
        _sum += tl.sum(x_chunk, axis=0)
        _sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    mean = _sum / num_cols
    # Var = E[x^2] - E[x]^2
    var = _sum_sq / num_cols - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # --- PASS 2: normalize and write ---
    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols
        x_chunk = tl.load(row_start_x + col_idxs, mask=mask, other=0.0)
        out = (x_chunk - mean) * rstd
        tl.store(row_start_y + col_idxs, out, mask=mask)


def triton_layernorm(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    rows, cols = x.shape
    y = torch.empty_like(x)
    triton_layernorm_kernel[(rows,)](
        x, y,
        x.stride(0), y.stride(0),
        cols,
        1e-5,
    )
    return y


# Benchmark
def benchmark_layernorm():
    dim = 4096
    print(f"\n--- Benchmarking LayerNorm Variants (dim={dim}x{dim}) ---")

    check_equal(pytorch_layernorm, manual_layernorm, dim=1024)
    check_equal(pytorch_layernorm, triton_layernorm, dim=1024)

    benchmark("1. Manual LayerNorm", run_operation_unary(dim, manual_layernorm))
    benchmark("2. PyTorch Native  ", run_operation_unary(dim, pytorch_layernorm))
    benchmark("3. Triton LayerNorm", run_operation_unary(dim, triton_layernorm))


if __name__ == "__main__":
    benchmark_layernorm()
