import torch

print("Torch version:", torch.__version__)
print("CUDA version (compiled):", torch.version.cuda)
print("CUDA available on this system:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
