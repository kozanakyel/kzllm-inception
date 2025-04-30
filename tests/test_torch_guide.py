import pytest
import torch


def test_cuda_installation():
    # Check if CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available. Please check your installation."

    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")  # Assuming at least one CUDA device
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Please check your installation.")
