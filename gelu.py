from utils import get_device, check_equal , run_benchmark
from triton.language.extra.cuda import libdevice
import torch
import triton
import triton.language as tl


# 1. Manual Implementation
def manual_gelu(x):
    # Memory Bound: Reads/Writes to HBM ~5 times
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

# 2. PyTorch Native
def pytorch_gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")

# 3. Triton Implementation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['num_elements'],
)
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute GeLU tanh approximation
    # sqrt(2/pi) constant
    k0 = 0.7978845608028654
    k1 = 0.044715

    x3 = x * x * x
    inner = k0 * (x + k1 * x3)

    # Correct Triton call
    tanh_val = libdevice.tanh(inner)

    y = 0.5 * x * (1.0 + tanh_val)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_gelu(x):
    x = x.contiguous()
    y = torch.empty_like(x)

    num_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    triton_gelu_kernel[grid](
        x, y,
        num_elements,
    )
    return y


# 4. Torch Compile
# Torch compile will effectively generate a fused kernel similar to our Triton one
compiled_gelu = torch.compile(manual_gelu)

def benchmark_gelu():
    # INCREASED DIMENSION to 4096 (approx 67MB of data)
    # This forces the GPU to access main memory, exposing bandwidth limits.
    dim = 4096 * 2 
    print(f"--- Benchmarking GeLU Variants (dim={dim}x{dim}) ---")
    
    # Correctness
    check_equal(manual_gelu, pytorch_gelu, dim=1024)
    check_equal(manual_gelu, triton_gelu, dim=1024)
    
    # Data Setup
    x = torch.randn(dim, dim, device=get_device())
    
    # Compile Warmup
    _ = compiled_gelu(x) 

    # Run
    run_benchmark("1. Manual Python", lambda x: manual_gelu(x), x)
    run_benchmark("2. PyTorch Native", lambda x: pytorch_gelu(x), x)
    run_benchmark("3. Triton Custom ", lambda x: triton_gelu(x), x)
    run_benchmark("4. Torch Compile ", lambda x: compiled_gelu(x), x)

if __name__ == "__main__":
    benchmark_gelu()