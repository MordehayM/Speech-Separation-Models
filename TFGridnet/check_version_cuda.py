import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Get the CUDA version used by PyTorch
    pytorch_cuda_version = torch.version.cuda
    print(f"PyTorch is using CUDA Version: {pytorch_cuda_version}")
else:
    print("CUDA is not available on this system.")
