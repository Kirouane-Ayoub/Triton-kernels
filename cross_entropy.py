import torch
import triton
import triton.language as tl
from utils import get_device, benchmark


# 1. PyTorch Native
def pytorch_cross_entropy(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets, reduction='none')


# 2. Manual Cross-Entropy
def manual_cross_entropy(logits, targets):
    # Step 1: numerically stable log-softmax (row-wise)
    max_logits = logits.max(dim=-1, keepdim=True)[0]
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - log_sum_exp

    # Step 2: gather the log-prob at the target index for each row
    # This is the negative log-likelihood
    loss = -log_probs[torch.arange(logits.shape[0], device=logits.device), targets]
    return loss


# 3. Triton Cross-Entropy
# Key idea: fuse log-softmax + target gather into a single kernel.
# Each program handles one row (one sample in the batch).
# This avoids materializing the full log-softmax output to HBM —
# we only write the scalar loss per row.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['num_classes'],
)
@triton.jit
def triton_cross_entropy_kernel(
    logits_ptr, targets_ptr, loss_ptr,
    logits_row_stride,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx * logits_row_stride
    offs = tl.arange(0, BLOCK_SIZE)

    # Load the target class for this row
    target = tl.load(targets_ptr + row_idx)

    # --- Single-pass online log-sum-exp (fused max + sum) ---
    # Same online trick as softmax: track running max and correct
    # the denominator on the fly when a new max is found.
    row_max = -float('inf')
    sum_exp = 0.0
    for col_start in range(0, num_classes, BLOCK_SIZE):
        col_idxs = col_start + offs
        mask = col_idxs < num_classes
        logits_chunk = tl.load(row_start + col_idxs, mask=mask, other=-float('inf'))
        chunk_max = tl.max(logits_chunk, axis=0)
        new_max = tl.maximum(row_max, chunk_max)
        # Rescale running sum for new max, then add new terms
        sum_exp = sum_exp * tl.exp(row_max - new_max) + tl.sum(tl.exp(logits_chunk - new_max), axis=0)
        row_max = new_max

    log_sum_exp = tl.log(sum_exp) + row_max

    # Load the logit at the target index
    target_logit = tl.load(row_start + target)

    # loss = -log_softmax[target] = -(target_logit - log_sum_exp)
    loss = log_sum_exp - target_logit

    tl.store(loss_ptr + row_idx, loss)


def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert logits.ndim == 2 and logits.is_cuda
    assert targets.ndim == 1 and targets.shape[0] == logits.shape[0]
    batch_size, num_classes = logits.shape
    loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)

    triton_cross_entropy_kernel[(batch_size,)](
        logits, targets, loss,
        logits.stride(0),
        num_classes,
    )
    return loss


# Benchmark
def benchmark_cross_entropy():
    batch_size = 4096
    num_classes = 32000  # typical vocab size for LLMs
    print(f"\n--- Benchmarking Cross-Entropy (batch={batch_size}, classes={num_classes}) ---")

    logits = torch.randn(batch_size, num_classes, device=get_device())
    targets = torch.randint(0, num_classes, (batch_size,), device=get_device())

    # Correctness
    ref = pytorch_cross_entropy(logits, targets)
    manual = manual_cross_entropy(logits, targets)
    triton_out = triton_cross_entropy(logits, targets)

    max_diff_manual = (ref - manual).abs().max().item()
    max_diff_triton = (ref - triton_out).abs().max().item()

    if max_diff_manual < 1e-3:
        print(f"  Manual vs PyTorch: PASS (max diff: {max_diff_manual:.2e})")
    else:
        print(f"  Manual vs PyTorch: FAIL (max diff: {max_diff_manual:.2e})")

    if max_diff_triton < 1e-3:
        print(f"  Triton vs PyTorch: PASS (max diff: {max_diff_triton:.2e})")
    else:
        print(f"  Triton vs PyTorch: FAIL (max diff: {max_diff_triton:.2e})")

    # Timing
    benchmark("1. PyTorch Native  ", lambda: pytorch_cross_entropy(logits, targets))
    benchmark("2. Manual Python   ", lambda: manual_cross_entropy(logits, targets))
    benchmark("3. Triton Fused    ", lambda: triton_cross_entropy(logits, targets))


if __name__ == "__main__":
    benchmark_cross_entropy()
