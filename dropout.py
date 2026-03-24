import torch
import triton
import triton.language as tl
from utils import get_device, benchmark


# 1. PyTorch Native
def pytorch_dropout(x, p=0.5):
    return torch.nn.functional.dropout(x, p=p, training=True)


# 2. Triton Dropout
# Key concept: instead of generating a random mask on CPU and sending it to GPU,
# we generate random numbers directly on-device using tl.rand().
# tl.rand() uses a seed + offset to produce deterministic pseudo-random numbers,
# which makes the kernel reproducible when given the same seed.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def triton_dropout_kernel(
    x_ptr, y_ptr,
    n_elements,
    p,        # drop probability
    inv_scale, # precomputed 1/(1-p) to avoid redundant division per program
    seed,     # random seed for reproducibility
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)

    # Generate random values in [0, 1) using seed + element offset
    random = tl.rand(seed, offs)

    # Keep elements where random >= p (i.e., drop with probability p)
    # Multiply by precomputed scale so expected value is preserved (inverted dropout)
    keep_mask = random >= p
    y = tl.where(keep_mask, x * inv_scale, 0.0)

    tl.store(y_ptr + offs, y, mask=mask)


def triton_dropout(x: torch.Tensor, p: float = 0.5, seed: int = 42) -> torch.Tensor:
    assert x.is_cuda
    assert 0.0 <= p < 1.0, "Drop probability must be in [0, 1)"
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    inv_scale = 1.0 / (1.0 - p)
    triton_dropout_kernel[grid](
        x, y,
        n_elements,
        p,
        inv_scale,
        seed,
    )
    return y


# Benchmark
def benchmark_dropout():
    dim = 4096 * 2
    p = 0.5
    print(f"\n--- Benchmarking Dropout Variants (dim={dim}x{dim}, p={p}) ---")

    # Correctness: we can't compare exact outputs (random), but we can check:
    # 1. Output has ~p fraction of zeros
    # 2. Non-zero values are scaled by 1/(1-p)
    x = torch.randn(1024, 1024, device=get_device())

    y_triton = triton_dropout(x, p=p, seed=123)
    zero_frac = (y_triton == 0).float().mean().item()
    print(f"Triton dropout zero fraction: {zero_frac:.3f} (expected ~{p})")

    # Check scaling: non-zero outputs should be x * 1/(1-p)
    nonzero_mask = y_triton != 0
    if nonzero_mask.any():
        scale = 1.0 / (1.0 - p)
        scaled_x = x[nonzero_mask] * scale
        max_diff = (y_triton[nonzero_mask] - scaled_x).abs().max().item()
        if max_diff < 1e-5:
            print(f"  Scaling check passed! (max diff: {max_diff:.2e})")
        else:
            print(f"  Scaling check FAILED (max diff: {max_diff:.2e})")

    # Reproducibility: same seed should give same output
    y1 = triton_dropout(x, p=p, seed=42)
    y2 = triton_dropout(x, p=p, seed=42)
    if torch.equal(y1, y2):
        print("  Reproducibility check passed!")
    else:
        print("  Reproducibility check FAILED")

    # Timing
    x_large = torch.randn(dim, dim, device=get_device())
    benchmark("1. PyTorch Dropout", lambda: pytorch_dropout(x_large, p=p))
    benchmark("2. Triton Dropout ", lambda: triton_dropout(x_large, p=p, seed=42))


if __name__ == "__main__":
    benchmark_dropout()
