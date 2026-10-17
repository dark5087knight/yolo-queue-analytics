"""
YOLOv8 Human Detection Training Script using PyTorch and CUDA.
Trains on Human Dataset v2 (Roboflow Universe) using the NVIDIA GeForce RTX 3050 GPU.
Automatically saves the trained weights to project/models/best.pt.
"""
import os
import sys
import shutil
import json
import argparse
from datetime import datetime
import torch
from ultralytics import YOLO

# Add parent directory (project root) to sys.path so config is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

DEFAULT_DATA_YAML = os.path.join(PROJECT_ROOT, "dataset.yaml")
DEFAULT_RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
LOCAL_BASE_MODEL = os.path.join(PROJECT_ROOT, "yolov8n.pt")

def train(
    data_yaml: str = DEFAULT_DATA_YAML,
    model_name: str = LOCAL_BASE_MODEL if os.path.exists(LOCAL_BASE_MODEL) else "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    workers: int = 2,
    patience: int = 10,
    device: str = "0",
):
    print("=" * 70)
    print("  YOLOv8 Queue Detection - Human Detector Training  ")
    print("=" * 70)

    # 1. Hardware & CUDA Verification
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device:      {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        selected_device = device
    else:
        print("WARNING: CUDA is not available. Training will run on CPU (very slow).")
        selected_device = "cpu"

    print(f"Dataset Config:   {data_yaml}")
    print(f"Base Weights:     {model_name}")
    print(f"Image Resolution: {imgsz}x{imgsz}")
    print(f"Batch Size:       {batch_size}")
    print(f"Max Epochs:       {epochs}")
    print(f"Early Stopping:   patience={patience}")
    print(f"Runs Output Dir:  {DEFAULT_RUNS_DIR}")
    print("=" * 70)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config file not found: {data_yaml}")

    # 2. Initialize Model
    print(f"\n[1/3] Loading base architecture: {model_name}...")
    model = YOLO(model_name)

    # 3. Start Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"human_yolov8_{timestamp}"

    print(f"\n[2/3] Launching GPU Training Run: {run_name}...")
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=selected_device,
        workers=workers,
        patience=patience,
        project=DEFAULT_RUNS_DIR,
        name=run_name,
        optimizer="auto",
        amp=True,  # Automatic Mixed Precision for RTX acceleration
        save=True,
        save_period=5,
        verbose=True,
    )

    # 4. Locate and Copy Best Weights
    best_pt_source = os.path.join(DEFAULT_RUNS_DIR, run_name, "weights", "best.pt")
    target_best_pt = os.path.join(config.MODELS_DIR, "best.pt")

    print(f"\n[3/3] Training finished. Exporting best weights...")
    if os.path.exists(best_pt_source):
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        shutil.copy2(best_pt_source, target_best_pt)
        print(f"Successfully saved best model weights to: {target_best_pt}")
    else:
        print(f"Warning: Could not find best.pt at {best_pt_source}")

    # 5. Validation on Validation Split
    print("\nRunning final validation on validation dataset...")
    val_metrics = model.val(data=data_yaml, device=selected_device, split="val")

    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    precision = float(val_metrics.box.mp)
    recall = float(val_metrics.box.mr)

    print("\n" + "=" * 50)
    print("          FINAL VALIDATION METRICS          ")
    print("=" * 50)
    print(f"mAP@0.50:       {map50:.4f}")
    print(f"mAP@0.50:0.95:  {map50_95:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print("=" * 50)

    # Save summary report JSON
    summary_data = {
        "date": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "epochs_completed": epochs,
        "batch_size": batch_size,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "best_weights": target_best_pt,
    }
    summary_file = os.path.join(config.MODELS_DIR, "training_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved summary metrics to: {summary_file}")

    return target_best_pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Human Dataset using GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 recommended for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--model", type=str, default=LOCAL_BASE_MODEL if os.path.exists(LOCAL_BASE_MODEL) else "yolov8n.pt", help="Base model weights")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to dataset.yaml")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
    )
