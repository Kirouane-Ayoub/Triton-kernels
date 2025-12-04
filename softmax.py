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


# 3. Triton Softmax
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['num_cols'],
)
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    """
    Chunked, numerically-stable softmax per row (one program per row).
    - BLOCK_SIZE: number of lanes per block/program (autotuned)
    """
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    # --- PASS 1: compute row-wise max (reduce over chunks) ---
    # initialize accumulator to very small value
    row_max = -1e9

    # sweep over columns in chunks
    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols

        x_ptrs = x_ptr + row_idx * x_row_stride + col_idxs
        # For max pass use a very negative "other" so masked lanes don't affect max.
        x_chunk = tl.load(x_ptrs, mask=mask, other=-1e9)

        # tl.max reduces across all lanes in the block for this vector and returns scalar
        chunk_max = tl.max(x_chunk, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # --- PASS 2: compute denominator (sum of exp(x - max)) ---
    denom = 0.0
    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols

        x_ptrs = x_ptr + row_idx * x_row_stride + col_idxs
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0)   # other doesn't matter here; masked lanes are ignored by mask

        # compute exp(x - max) in registers
        # Using tl.exp is fine; libdevice.exp also possible
        ex = tl.exp(x_chunk - row_max)

        # sum across lanes in this block's vector
        denom += tl.sum(ex, axis=0)

    # denom is a scalar (sum over all chunks)
    # --- PASS 3: final division and write result ---
    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_cols

        x_ptrs = x_ptr + row_idx * x_row_stride + col_idxs
        y_ptrs = y_ptr + row_idx * y_row_stride + col_idxs

        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0)
        ex = tl.exp(x_chunk - row_max)
        out = ex / denom
        tl.store(y_ptrs, out, mask=mask)


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