import torch
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print("--- Verification Successful ---")
print(f"PyTorch version: {torch.__version__}")
print(f"MMCV version: {mmcv.__version__}")
print(f"MMCV CUDA version: {get_compiling_cuda_version()}")
print(f"MMCV Compiler version: {get_compiler_version()}")