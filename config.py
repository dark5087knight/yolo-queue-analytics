"""
Configuration settings for the YOLOv8 + DeepSORT Queue Detection and Analytics System.
This file can be dynamically updated by the Wait Trainer tool.
"""
import os
from pathlib import Path
import torch

# Hardware & CUDA Settings
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0 if USE_CUDA else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if USE_CUDA else "CPU Only"
GPU_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if USE_CUDA else 0.0

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATASET_YAML_PATH = os.path.join(BASE_DIR, "dataset.yaml")

# Model settings
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
FALLBACK_MODEL_NAME = "yolov8n.pt"  # Pretrained fallback if best.pt is not yet trained
CONFIDENCE_THRESHOLD = 0.55         # 0.55 filters out low-confidence false positives and background clutter
IOU_THRESHOLD = 0.50                # NMS threshold preventing duplicate bounding boxes
TARGET_CLASSES = [0, 1]             # 'Human ' and 'Human' classes from dataset

# Multi-Object Tracking Settings (Deep SORT & ByteTrack)
TRACKER_TYPE = "deepsort"           # "deepsort" (Deep SORT with appearance vectors) or "bytetrack" (high-speed YOLO tracking)
SHOW_TRAJECTORIES = True            # Draw motion trajectory trail lines behind tracked persons
TRAJECTORY_LENGTH = 30              # Number of frames of history to draw in trajectory trail
MAX_COSINE_DISTANCE = 0.38          # 0.38 prevents false ID switches under pose variation
NN_BUDGET = 100                     # Feature gallery capacity per track
MAX_AGE = 60                        # 60 frames (2.0s) preserves customer track through temporary occlusions
MIN_HITS = 2                        # Require 2 consecutive frames to confirm track (eliminates 1-frame noise)

# Anti-Jitter & Geometry Noise Filters
MIN_PERSON_HEIGHT = 40              # Minimum bounding box height in pixels to filter out tiny noise artifacts
MIN_PERSON_AREA = 800               # Minimum bounding box area in pixels (width * height)
ENABLE_BOX_SMOOTHING = True         # Exponential moving average filter to eliminate box jitter/vibration
BOX_SMOOTHING_ALPHA = 0.65          # Smoothing factor (0.65 current + 0.35 history)

# Queue Thresholds & Status
ALERT_QUEUE_THRESHOLD = 5         # Queue length >= ALERT_QUEUE_THRESHOLD triggers ALERT
BUSY_QUEUE_THRESHOLD = 3          # Queue length >= BUSY_QUEUE_THRESHOLD triggers Busy
SMOOTHING_WINDOW = 4              # Number of frames for count smoothing

# Calibrated Operational Metrics (Updated by Wait Trainer)
AVERAGE_WAIT_TIME_SECONDS = 14.12 # Default estimated service time per customer in seconds
PEAK_HOUR = 8                     # Default peak operational hour (0-23)

# Video Processing & Playback
TARGET_FPS = 30                   # Target frame processing rate
REALTIME_PLAYBACK = True          # True = sync video playback to recorded FPS; False = maximum unthrottled GPU speed
MIN_FRAMES_IN_ZONE = 5            # Minimum consecutive frames before a person is confirmed in queue (avoids transients)
MIN_FRAMES_BEFORE_EXIT = 8         # Grace period to avoid flickering when person temporarily blocked
