#!/usr/bin/env python3
"""
Export trained YOLOv8 model for Raspberry Pi deployment
Optimized for cat_yolov8n.pt on Raspberry Pi 4 (4GB RAM)
"""
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def find_model():
    """Find cat_yolov8n.pt model automatically"""
    search_paths = [
        PROJECT_ROOT / "models" / "trained" / "cat_yolov8n.pt",
        PROJECT_ROOT / "cat_yolov8n.pt",
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def export_model(model_path: str, format: str = 'onnx', imgsz: int = 640, optimize_rpi: bool = True):
    """
    Export YOLOv8 model optimized for Raspberry Pi 4
    
    Args:
        model_path: Path to trained .pt model
        format: Export format (onnx, tflite, ncnn)
        imgsz: Input image size (smaller = faster on RPi)
        optimize_rpi: Apply Raspberry Pi optimizations
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Exporting Model for Raspberry Pi 4 (4GB RAM)")
    print(f"{'='*60}")
    print(f"Model:       {model_path.name}")
    print(f"Format:      {format.upper()}")
    print(f"Image Size:  {imgsz}x{imgsz}")
    print(f"Optimization: {'Enabled' if optimize_rpi else 'Disabled'}")
    print()
    
    try:
        # Load trained model
        print("[1/3] Loading model...")
        model = YOLO(str(model_path))
        print(f"      Model type: {model.model.__class__.__name__}")
        
        # Export based on format
        print(f"[2/3] Exporting to {format.upper()}...")
        
        export_args = {
            'format': format,
            'imgsz': imgsz,
        }
        
        # Format-specific optimizations for RPi
        if optimize_rpi:
            if format == 'onnx':
                export_args['simplify'] = True  # Simplify ONNX graph
                export_args['dynamic'] = False  # Static shapes for better performance
                
            elif format == 'tflite':
                export_args['int8'] = True  # INT8 quantization for speed
                
            elif format == 'ncnn':
                # NCNN is already optimized for ARM
                pass
        
        export_path = model.export(**export_args)
        
        print(f"[3/3] Export complete!")
        print()
        print(f"{'='*60}")
        print(f"SUCCESS! Model exported to:")
        print(f"  {export_path}")
        print(f"{'='*60}")
        print()
        
        # Performance estimates for RPi 4
        perf_estimates = {
            'onnx': '5-10 FPS',
            'tflite': '3-7 FPS (INT8: 5-10 FPS)',
            'ncnn': '8-12 FPS',
        }
        
        print("Expected Performance on Raspberry Pi 4:")
        print(f"  Estimated FPS: {perf_estimates.get(format, 'Unknown')}")
        print()
        
        # Transfer instructions
        print("Next Steps:")
        print(f"  1. Transfer to Raspberry Pi:")
        if Path(export_path).is_dir():
            print(f"     scp -r {export_path} pi@raspberrypi.local:~/smart-home-cat-detector/models/trained/")
        else:
            print(f"     scp {export_path} pi@raspberrypi.local:~/smart-home-cat-detector/models/trained/")
        print()
        print(f"  2. Update .env on RPi:")
        print(f"     MODEL_PATH=models/trained/{Path(export_path).name}")
        print()
        print(f"  3. Run on Raspberry Pi:")
        print(f"     export HEADLESS_MODE=1")
        print(f"     python3 main.py live")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_all_formats(model_path: str, imgsz: int = 640):
    """
    Export model to all recommended formats for Raspberry Pi
    """
    formats = ['onnx', 'ncnn', 'tflite']
    
    print(f"\n{'='*60}")
    print(f"Exporting to ALL formats for Raspberry Pi")
    print(f"{'='*60}\n")
    
    results = {}
    for fmt in formats:
        print(f"\n--- Exporting to {fmt.upper()} ---\n")
        success = export_model(model_path, format=fmt, imgsz=imgsz)
        results[fmt] = success
        
        if not success:
            print(f"[WARNING] {fmt.upper()} export failed, continuing...\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    for fmt, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {fmt.upper()}")
    print()
    
    successful = [fmt for fmt, success in results.items() if success]
    if successful:
        print(f"Successfully exported: {', '.join([f.upper() for f in successful])}")
        print()
        print("Recommendation for RPi 4:")
        if 'ncnn' in successful:
            print("  → Use NCNN for best performance (fastest)")
        elif 'onnx' in successful:
            print("  → Use ONNX for good balance (compatible & fast)")
        elif 'tflite' in successful:
            print("  → Use TFLite INT8 (smallest size)")
    
    return all(results.values())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export cat_yolov8n.pt for Raspberry Pi 4 deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX (recommended)
  python src/models/export_model.py
  
  # Export to NCNN (fastest on RPi)
  python src/models/export_model.py --format ncnn
  
  # Export to TFLite with INT8 quantization
  python src/models/export_model.py --format tflite
  
  # Export to all formats
  python src/models/export_model.py --all
  
  # Smaller size for faster inference (trade accuracy for speed)
  python src/models/export_model.py --imgsz 416
  
Recommended for RPi 4 (4GB):
  - Format: ONNX or NCNN
  - Image Size: 640 (default) or 416 (faster)
  - Expected: 5-12 FPS depending on format
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model (default: auto-detect cat_yolov8n.pt)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['onnx', 'tflite', 'ncnn'],
        default='onnx',
        help='Export format (default: onnx - recommended for RPi)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640, use 416 for faster inference)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Export to all formats (onnx, tflite, ncnn)'
    )
    
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable Raspberry Pi optimizations'
    )
    
    args = parser.parse_args(argv)
    
    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        print("Searching for cat_yolov8n.pt model...")
        model_path = find_model()
    
    if model_path is None or not model_path.exists():
        print(f"[ERROR] Model not found!")
        print(f"\nSearched locations:")
        print(f"  - models/trained/cat_yolov8n.pt")
        print(f"  - cat_yolov8n.pt")
        print(f"  - runs/detect/train/weights/best.pt")
        print(f"\nPlease train a model first: python main.py train")
        print(f"Or specify path: python src/models/export_model.py --model path/to/model.pt")
        sys.exit(1)
    
    print(f"Found model: {model_path}")
    
    # Export
    if args.all:
        success = export_all_formats(str(model_path), args.imgsz)
    else:
        success = export_model(
            str(model_path), 
            args.format, 
            args.imgsz,
            optimize_rpi=not args.no_optimize
        )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])