# import tensorflow as tf
import torch
import os
import sys

def check_pytorch_gpu():
    """
    Check if PyTorch can access GPU
    """
    print("\n=== PyTorch GPU Configuration ===")
    
    # Check PyTorch version
    print("\nPyTorch Version:", torch.__version__)
    
    # Check if CUDA is available
    print("\nCUDA Available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        # Get the current device
        current_device = torch.cuda.current_device()
        print("Current CUDA Device:", current_device)
        
        # Get the name of the current device
        print("Device Name:", torch.cuda.get_device_name(current_device))
        
        # Get the number of available GPUs
        print("Number of GPUs Available:", torch.cuda.device_count())
        
        # Test PyTorch GPU computation
        print("\n=== Testing PyTorch GPU Access ===")
        try:
            # Create tensors on GPU
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).cuda()
            b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).cuda()
            
            # Perform matrix multiplication
            c = torch.mm(a, b)
            print("\nMatrix multiplication test successful on GPU")
            print("Result:", c)
            
            # Get memory statistics
            print("\n=== GPU Memory Info ===")
            print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached GPU Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
        except RuntimeError as e:
            print("\nError in PyTorch GPU computation:", e)
    else:
        print("\nNo GPU available for PyTorch. Using CPU only.")

if __name__ == "__main__":
    check_pytorch_gpu()
