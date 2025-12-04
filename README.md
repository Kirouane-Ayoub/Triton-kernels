# My Triton Kernels

This repository is a part of my Triton learning journey. I will be adding more kernels, fixes, and optimizations as I explore Triton.

## Benchmarking Results

| Kernel | Dimensions | Variant | Time (ms) |
| :--- | :--- | :--- | :--- |
| **ReLU** | 16384x16384 | Manual PyTorch | 18.1418 |
| | | Triton Custom | 9.0093 |
| **GeLU** | 8192x8192 | Manual Python | 25.0571 |
| | | PyTorch Native | 2.3076 |
| | | Triton Custom | 2.3011 |
| | | Torch Compile | 2.1688 |
| **Softmax** | 4096x4096 | Manual | 2.2748 |
| | | PyTorch Native | 0.5678 |
| | | Triton | 0.9235 |
| **MatMul** | 2048x2048 | PyTorch Native | 5.0519 |
| | | Triton Custom | 5.0077 |
