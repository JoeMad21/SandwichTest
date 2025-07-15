import os


def setup_cuda_debugging():
    """Set CUDA_LAUNCH_BLOCKING to help debug CUDA errors."""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("🔧 Set CUDA_LAUNCH_BLOCKING=1 for debugging")
