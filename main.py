#!/usr/bin/env python3
"""
Smart Home Cat Detector - Central CLI
Simplified interface for all project functions
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def collect_data():
    """Start data collection from RTSP camera"""
    from src.data_collection.collect_data import main
    print("=== Data Collection Started ===")
    main()


def auto_label():
    """Automatically label collected images"""
    from src.labeling.auto_label import main
    print("=== Auto-Labeling Started ===")
    main()


def train():
    """Train YOLOv8 model"""
    from src.models.trainer import main
    print("=== Training Started ===")
    main()


def validate():
    """Validate trained model"""
    from src.benchmark.val import main
    print("=== Validation Started ===")
    main()


def benchmark():
    """Run performance benchmark"""
    from src.benchmark.benchmark import main
    print("=== Benchmark Started ===")
    main()


def live_detect(debug=False):
    """Run live detection on RTSP stream"""
    from src.live_monitoring.start_monitoring import main
    print("=== Live Detection Started ===")
    main(debug=debug)


def export_model():
    """Export model for deployment (Raspberry Pi)"""
    from src.models.export_model import main
    print("=== Model Export Started ===")
    main([])  # Pass empty list to use defaults


def main():
    parser = argparse.ArgumentParser(
        description="Smart Home Cat Detector - Central Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect      # Collect data from RTSP stream
  python main.py label        # Auto-label collected images
  python main.py train        # Train the model
  python main.py validate     # Validate the model
  python main.py benchmark    # Run performance benchmark
  python main.py export       # Export model for Raspberry Pi
  python main.py live         # Start live detection
  python main.py live --debug # Test webhook without live stream

For full documentation see README.md
        """
    )
    
    parser.add_argument(
        'command',
        choices=['collect', 'label', 'train', 'validate', 'benchmark', 'export', 'live'],
        help='Function to execute'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (for live command: test webhook without stream)'
    )
    
    args = parser.parse_args()
    
    commands = {
        'collect': collect_data,
        'label': auto_label,
        'train': train,
        'validate': validate,
        'benchmark': benchmark,
        'export': export_model,
        'live': lambda: live_detect(debug=args.debug)
    }
    
    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
