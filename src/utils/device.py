"""
Device selection utilities
"""
import torch


def get_device(prefer_gpu: bool = True) -> str:
    """
    Get the appropriate device for PyTorch/YOLO operations
    
    Args:
        prefer_gpu: If True and CUDA is available, return GPU device
        
    Returns:
        Device string: '0' for GPU, 'cpu' for CPU
    """
    if prefer_gpu and torch.cuda.is_available():
        return '0'
    return 'cpu'


def get_device_info() -> dict:
    """
    Get detailed device information
    
    Returns:
        Dictionary with device details
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': get_device(),
        'device_name': None,
        'device_count': 0
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_count'] = torch.cuda.device_count()
    
    return info


def print_device_info():
    """Print device information to console"""
    info = get_device_info()
    print("\n=== Device Information ===")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Selected Device: {info['device']}")
    if info['cuda_available']:
        print(f"GPU Name: {info['device_name']}")
        print(f"GPU Count: {info['device_count']}")
    print("=" * 25 + "\n")
