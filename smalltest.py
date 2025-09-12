# MINIMAL GPU TEST - Run this first to diagnose the issue
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test GPU tensor creation and operation
    device = torch.device('cuda')
    x_gpu = torch.randn(1000, 1000, device=device)
    y_gpu = torch.randn(1000, 1000, device=device)
    z_gpu = torch.mm(x_gpu, y_gpu)  # Matrix multiplication on GPU
    print(f"GPU tensor device: {z_gpu.device}")
    print("✅ GPU test successful!")
else:
    print("❌ CUDA not available")