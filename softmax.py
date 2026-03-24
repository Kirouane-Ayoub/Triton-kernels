from utils import run_operation_unary, check_equal , benchmark
import torch
import triton
import triton.language as tl

def manual_softmax(x):
    # Multiple reads/writes (bad for memory bandwidth)
    x_max = x.max(dim=1)[0]
    x = x - x_max[:, None]
    numerator = torch.exp(x)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]


def pytorch_softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)


# 3. Triton Softmax (Online algorithm — 2 passes instead of 3)
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
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    """
    Online softmax: fuses max and sum into a single pass over data.
    Pass 1: compute max and denominator together (online algorithm).
    Pass 2: normalize and write output.
    This reads global memory 2x instead of 3x.
    """
    row_idx = tl.program_id(0)
    row_start_x = x_ptr + row_idx * x_row_stride
    row_start_y = y_ptr + row_idx * y_row_stride
    offs = tl.arange(0, BLOCK_SIZE)

    # --- PASS 1: online softmax — fused max + sum ---
    # Track running max and a denominator that gets corrected on the fly.
    row_max = -float('inf')
    denom = 0.0

    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols
        x_chunk = tl.load(row_start_x + col_idxs, mask=mask, other=-float('inf'))

        # New max for this chunk
        chunk_max = tl.max(x_chunk, axis=0)
        new_max = tl.maximum(row_max, chunk_max)

        # Correct the running denominator for the new max, then add new terms.
        # denom was sum(exp(x_i - old_max)), rescale to sum(exp(x_i - new_max))
        denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x_chunk - new_max), axis=0)
        row_max = new_max

    # --- PASS 2: normalize and write ---
    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols
        x_chunk = tl.load(row_start_x + col_idxs, mask=mask, other=-float('inf'))
        out = tl.exp(x_chunk - row_max) / denom
        tl.store(row_start_y + col_idxs, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    rows, cols = x.shape
    y = torch.empty_like(x)
    # Launch one program per row; BLOCK_SIZE chosen by autotuner
    triton_softmax_kernel[(rows,)](
        x, y,
        x.stride(0), y.stride(0),
        cols,
    )
    return y



# Benchmark Softmax
def benchmark_softmax():
    dim = 4096
    print(f"\n--- Benchmarking Softmax Variants (dim={dim}x{dim}) ---")

    # Correctness
    check_equal(pytorch_softmax, manual_softmax, dim=1024)
    check_equal(pytorch_softmax, triton_softmax, dim=1024)

    # Timing
    benchmark("1. Manual Softmax", run_operation_unary(dim, manual_softmax))
    benchmark("2. PyTorch Native", run_operation_unary(dim, pytorch_softmax))
    benchmark("3. Triton Softmax", run_operation_unary(dim, triton_softmax))

benchmark_softmax()