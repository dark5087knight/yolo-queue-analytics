# YOLOv8 + DeepSORT Queue Detection & Analytics System

Desktop queue management and monitoring application developed using **Python**, **YOLOv8** (PyTorch / CUDA), **Deep SORT**, and **PyQt5**.

---

## Project Structure

```
project/
├── config.py                 # Tunable and calibrated operational parameters
├── dataset.yaml              # Dataset path and class specifications
├── deep_sort_tracker.py      # Multi-person Deep SORT tracking module
├── queue_manager.py          # ROI filtering, wait timing, and smoothing logic
├── detection_thread.py       # QThread handling frame capture & GPU inference
├── main_window.py            # Main PyQt5 dashboard window with interactive ROI
├── report_window.py          # Queue Analytics Report window (CSV & PDF export)
├── wait_trainer.py           # Calibration utility updating config.py
├── main.py                   # Application entry point
├── demo_queue.mp4            # Sample queue video for testing
├── training/                 # Training scripts
│   └── train_model.py        # YOLOv8 model training script using PyTorch & CUDA
├── models/                   # Stores trained weights (best.pt) and metrics
├── reports/                  # Stores exported CSV logs and PDF reports
├── runs/                     # Training run logs and checkpoints
└── utils/
    └── create_demo_video.py  # Utility to create test queue videos
```

---

## Quick Start Guide

### 1. Launch the Main Application
```bash
cd project
python main.py
```
- Click **Select Video** and choose `demo_queue.mp4` (or any video), or click **Camera** for webcam.
- Draw your queue zone:
  - Click **Zone (Drag)** to click and drag a rectangle on the video.
  - Or click **Zone (4 Pts)** to click 4 points forming a polygon.
- Click **Start** to run real-time queue detection, tracking, and wait-time estimation!
- Click **Generate Full Report** to view summary statistics, customer logs, and export to CSV or PDF.

---

### 2. Train the YOLOv8 Human Detector on GPU
To train the detection model on the provided dataset (`Human Dataset v2`) using your NVIDIA GeForce RTX 3050 GPU with CUDA:

```bash
cd project
python training/train_model.py --epochs 30 --batch 16 --device 0
```
- Uses Automatic Mixed Precision (AMP) for RTX acceleration.
- Automatically saves the best model weights to `models/best.pt`.
- Saves validation metrics (mAP50, mAP50-95, precision, recall) to `models/training_summary.json`.

---

### 3. Launch the Wait-Time Trainer (Config Calibration)
To calibrate the system's operational parameters based on logged history:
```bash
cd project
python wait_trainer.py
```
- Click **Select CSV** and choose `reports/sample_queue_data.csv` (or an exported queue report).
- Click **Start training — apply to config.py**.
- The tool calculates empirical average service wait and peak traffic hour, and updates `config.py` directly.
