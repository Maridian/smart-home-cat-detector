import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ultralytics import YOLO

MODEL_PATH = PROJECT_ROOT / "cat_yolov8n.pt"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS = 10
BENCHMARK_RUNS = 50
BATCH_SIZES = [1, 2, 4, 8, 16]

def run_benchmark(device_str: str):
    """Measures latency and FPS across different batch sizes for a given device."""
    if not MODEL_PATH.exists():
        print(f"[Error] Model weights not found at: {MODEL_PATH}")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))
    results = {"batch_sizes": BATCH_SIZES, "latencies_ms": [], "fps": []}

    print(f"\n--- Running Benchmark on: {device_str.upper()} ---")

    for batch_size in BATCH_SIZES:
        # Dummy image tensor matching YOLO input shape (Batch, Channels, Height, Width)
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_batch = [dummy_input] * batch_size

        # Warmup iterations to initialize CUDA/Torch drivers
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_batch, device=device_str, verbose=False)

        # Timed benchmark loop
        start_time = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = model(dummy_batch, device=device_str, verbose=False)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_images = BENCHMARK_RUNS * batch_size
        
        avg_latency_ms = (total_time / BENCHMARK_RUNS) * 1000
        fps = total_images / total_time

        results["latencies_ms"].append(avg_latency_ms)
        results["fps"].append(fps)

        print(f"Batch Size: {batch_size:2d} | Avg Latency: {avg_latency_ms:6.2f} ms | Throughput: {fps:6.2f} FPS")

    return results

def generate_benchmark_plots(cpu_res, gpu_res=None):
    """Generates and saves performance visualization plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Latency (ms) ---
    ax1.plot(cpu_res["batch_sizes"], cpu_res["latencies_ms"], marker="o", linewidth=2, label="CPU", color="#d95f02")
    if gpu_res:
        ax1.plot(gpu_res["batch_sizes"], gpu_res["latencies_ms"], marker="s", linewidth=2, label="GPU (CUDA)", color="#7570b3")
    
    ax1.set_title("Inference Latency per Batch (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(BATCH_SIZES)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Plot 2: Throughput (FPS) ---
    ax2.plot(cpu_res["batch_sizes"], cpu_res["fps"], marker="o", linewidth=2, label="CPU", color="#d95f02")
    if gpu_res:
        ax2.plot(gpu_res["batch_sizes"], gpu_res["fps"], marker="s", linewidth=2, label="GPU (CUDA)", color="#7570b3")

    ax2.set_title("Throughput / FPS (Higher is Better)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Frames Per Second (FPS)")
    ax2.set_xticks(BATCH_SIZES)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = EXPORTS_DIR / "benchmark_results.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[✓] Benchmark chart successfully saved to: {output_path}")

def main():
    cpu_results = run_benchmark(device_str="cpu")
    
    gpu_results = None
    if torch.cuda.is_available():
        gpu_results = run_benchmark(device_str="0")
    else:
        print("\n[Info] CUDA GPU not available. Skipping GPU benchmark.")

    generate_benchmark_plots(cpu_results, gpu_results)

if __name__ == "__main__":
    main()