"""
Detection Thread Module.
Runs video capture, YOLOv8 GPU inference, Deep SORT tracking, and Queue logic
in a dedicated PyQt5 QThread to keep the GUI responsive.
"""
import os
import time
from typing import Optional, List, Tuple
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

import config
from deep_sort_tracker import DeepSortTracker
from queue_manager import QueueManager, format_duration

class DetectionThread(QThread):
    # Signals emitted to the main GUI
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(dict)
    fps_ready = pyqtSignal(float)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        source: str,
        queue_manager: QueueManager,
        model_path: Optional[str] = None,
        conf_threshold: float = config.CONFIDENCE_THRESHOLD,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.queue_manager = queue_manager
        self.model_path = model_path or config.DEFAULT_MODEL_PATH
        self.conf_threshold = conf_threshold

        self.running = False
        self.paused = False
        self.tracker = DeepSortTracker()
        self.model = None
        self.track_history = defaultdict(lambda: deque(maxlen=getattr(config, "TRAJECTORY_LENGTH", 30)))
        self.smoothed_boxes = {}

    def load_model(self):
        """Loads and pre-warms YOLOv8 and tracking models on GPU so playback starts instantly without freezing."""
        # Use trained model if exists, otherwise fallback to yolov8n.pt
        actual_path = self.model_path
        if not os.path.exists(actual_path):
            actual_path = config.FALLBACK_MODEL_NAME
            self.status_message.emit(f"Trained model not found at {self.model_path}. Using {actual_path}...")

        try:
            self.model = YOLO(actual_path)
            if torch.cuda.is_available():
                self.model.to("cuda:0")
                # Pre-warm GPU detection and ByteTrack graphs
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                with torch.inference_mode():
                    self.model.predict(dummy, device=0, verbose=False)
                    try:
                        self.model.track(dummy, device=0, tracker="bytetrack.yaml", verbose=False)
                    except Exception:
                        pass
                gpu_name = torch.cuda.get_device_name(0)
                self.status_message.emit(f"GPU Accelerated: {gpu_name} (CUDA:0)")
            else:
                self.status_message.emit("Running on CPU (CUDA not detected)")

            # Pre-warm Deep SORT MobileNet embedder and pre-allocate CUDA memory
            self.tracker.warmup()
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model on GPU: {str(e)}")
            self.model = None

    def run(self):
        self.load_model()
        if self.model is None:
            return

        # Handle numeric webcam index vs string file path
        is_video_file = not str(self.source).isdigit()
        if is_video_file:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
        else:
            cap_idx = int(self.source)
            cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cap_idx)

        if not cap.isOpened():
            self.error_occurred.emit(f"Could not open video source: {self.source}")
            return

        self.running = True
        fps_counter = 0
        fps_timer = time.time()
        last_stats_emit = 0.0
        current_fps = 0.0

        device = 0 if torch.cuda.is_available() else "cpu"

        # Color palette for tracks (cycling colors)
        color_palette = [
            (255, 99, 71),   # Tomato
            (50, 205, 50),   # LimeGreen
            (30, 144, 255),  # DodgerBlue
            (255, 215, 0),   # Gold
            (186, 85, 211),  # MediumOrchid
            (0, 255, 255),   # Cyan
            (255, 105, 180), # HotPink
            (255, 140, 0),   # DarkOrange
        ]

        # Determine source playback rate
        if is_video_file:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = raw_fps if (raw_fps and raw_fps > 0 and not np.isnan(raw_fps)) else 30.0
        else:
            target_fps = 60.0

        min_height = getattr(config, "MIN_PERSON_HEIGHT", 40)
        min_area = getattr(config, "MIN_PERSON_AREA", 800)
        enable_smoothing = getattr(config, "ENABLE_BOX_SMOOTHING", True)
        smooth_alpha = getattr(config, "BOX_SMOOTHING_ALPHA", 0.65)

        while self.running:
            if self.paused:
                self.msleep(50)
                continue

            start_tick = time.time()
            ret, frame = cap.read()
            if not ret:
                # If file, loop back to beginning
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            has_zones = len(self.queue_manager.zones) > 0
            active_tracks = []

            # 1. Resolve target classes dynamically from model names
            target_classes = [
                idx for idx, name in self.model.names.items()
                if any(k in name.lower() for k in ["person", "human"])
            ]
            if not target_classes:
                target_classes = [0]

            # 2. Strict Zone-Only Multi-Object Tracking Engine (ByteTrack or Deep SORT)
            # If no zone has been defined yet, do NOT detect, track, or count anyone!
            if has_zones:
                tracker_type = getattr(config, "TRACKER_TYPE", "deepsort").lower()

                if tracker_type == "bytetrack":
                    with torch.inference_mode():
                        results = self.model.track(
                            frame,
                            persist=True,
                            conf=self.conf_threshold,
                            classes=target_classes,
                            device=device,
                            tracker="bytetrack.yaml",
                            verbose=False,
                            imgsz=640,
                        )
                    if results and len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        for box in boxes:
                            if box.id is not None:
                                try:
                                    track_id = int(box.id[0].cpu().numpy())
                                except Exception:
                                    track_id = 1
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                                # Anti-Noise Geometric Validation
                                bw = x2 - x1
                                bh = y2 - y1
                                if bh < min_height or (bw * bh) < min_area:
                                    continue
                                aspect = bw / max(1.0, float(bh))
                                if aspect < 0.15 or aspect > 2.5:
                                    continue

                                # Strict Filter: ONLY keep people inside defined queue zone
                                is_in, _ = self.queue_manager.is_box_in_zone(x1, y1, x2, y2)
                                if not is_in:
                                    if track_id in self.track_history:
                                        del self.track_history[track_id]
                                    if track_id in self.smoothed_boxes:
                                        del self.smoothed_boxes[track_id]
                                    continue
                                active_tracks.append((track_id, (x1, y1, x2, y2), conf))
                else:
                    with torch.inference_mode():
                        results = self.model.predict(
                            frame,
                            conf=self.conf_threshold,
                            classes=target_classes,
                            device=device,
                            verbose=False,
                            imgsz=640,
                        )
                    detections = []
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = self.model.names.get(cls_id, "Human")
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                            # Anti-Noise Geometric Validation
                            bw = x2 - x1
                            bh = y2 - y1
                            if bh < min_height or (bw * bh) < min_area:
                                continue
                            aspect = bw / max(1.0, float(bh))
                            if aspect < 0.15 or aspect > 2.5:
                                continue

                            # PRE-FILTER: Discard people outside zone BEFORE expensive appearance embedding
                            is_in, _ = self.queue_manager.is_box_in_zone(x1, y1, x2, y2)
                            if not is_in:
                                continue
                            detections.append((x1, y1, x2, y2, conf, cls_name))

                    raw_tracks = self.tracker.update(detections, frame)

                    # POST-FILTER: Discard any Kalman-predicted tracks that moved outside the zone
                    for track_id, (x1, y1, x2, y2), conf in raw_tracks:
                        is_in, _ = self.queue_manager.is_box_in_zone(x1, y1, x2, y2)
                        if not is_in:
                            if track_id in self.track_history:
                                del self.track_history[track_id]
                            if track_id in self.smoothed_boxes:
                                del self.smoothed_boxes[track_id]
                            continue
                        active_tracks.append((track_id, (x1, y1, x2, y2), conf))

                # Apply Exponential Moving Average (EMA) Box Smoothing to eliminate visual jitter
                smoothed_tracks = []
                for track_id, (x1, y1, x2, y2), conf in active_tracks:
                    if enable_smoothing:
                        if track_id in self.smoothed_boxes:
                            px1, py1, px2, py2 = self.smoothed_boxes[track_id]
                            x1 = int(smooth_alpha * x1 + (1.0 - smooth_alpha) * px1)
                            y1 = int(smooth_alpha * y1 + (1.0 - smooth_alpha) * py1)
                            x2 = int(smooth_alpha * x2 + (1.0 - smooth_alpha) * px2)
                            y2 = int(smooth_alpha * y2 + (1.0 - smooth_alpha) * py2)
                        self.smoothed_boxes[track_id] = (x1, y1, x2, y2)
                    smoothed_tracks.append((track_id, (x1, y1, x2, y2), conf))
                active_tracks = smoothed_tracks

                # 3. Record Trajectory History (only for active tracks strictly inside zone)
                current_tids = {tid for tid, _, _ in active_tracks}
                for tid in list(self.track_history.keys()):
                    if tid not in current_tids:
                        del self.track_history[tid]
                for tid in list(self.smoothed_boxes.keys()):
                    if tid not in current_tids:
                        del self.smoothed_boxes[tid]

                for track_id, (x1, y1, x2, y2), conf in active_tracks:
                    cx = (x1 + x2) // 2
                    cy = y2
                    self.track_history[track_id].append((cx, cy))
            else:
                # No zones defined: reset tracks and history
                active_tracks = []
                self.track_history.clear()
                self.smoothed_boxes.clear()

            # 4. Update Queue Manager Logic and Emit Stats at 4 Hz (decoupled from video FPS)
            stats = self.queue_manager.update(active_tracks)
            now = time.time()
            if now - last_stats_emit >= 0.25:
                self.stats_ready.emit(stats)
                last_stats_emit = now

            # 5. Draw Visual Overlays on Frame
            annotated_frame = frame.copy()

            # A. Draw Queue Zones (bold red outline matching gui_04)
            for zone in self.queue_manager.zones:
                pts = zone.polygon.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
                if len(zone.points) > 0:
                    lx, ly = zone.points[0]
                    cv2.putText(
                        annotated_frame, f"Zone {zone.zone_id}", (lx, max(22, ly - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

            if not has_zones:
                # Prominent helper banner when no zone is drawn yet
                fh, fw = annotated_frame.shape[:2]
                banner_txt = "Define a Queue Zone (Click 'Drag' or '4-Pts') to begin tracking"
                (tw, th), _ = cv2.getTextSize(banner_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                bx = (fw - tw) // 2
                by = 40
                cv2.rectangle(annotated_frame, (bx - 12, by - th - 10), (bx + tw + 12, by + 10), (15, 23, 42), -1)
                cv2.rectangle(annotated_frame, (bx - 12, by - th - 10), (bx + tw + 12, by + 10), (56, 189, 248), 2)
                cv2.putText(annotated_frame, banner_txt, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

            # B. Draw Motion Trajectory Trails (Only for people currently inside the zone)
            if getattr(config, "SHOW_TRAJECTORIES", True):
                for track_id, points in list(self.track_history.items()):
                    if len(points) > 1:
                        color = color_palette[track_id % len(color_palette)]
                        pts = np.array(list(points), np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
                        cv2.circle(annotated_frame, points[-1], 3, color, -1)

            # C. Draw Bounding Boxes and Dual-Pill Individual Timers (Only for people inside zone)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thick = 2

            for track_id, (x1, y1, x2, y2), conf in active_tracks:
                color = color_palette[track_id % len(color_palette)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Format wait time tag if currently inside queue zone
                if track_id in self.queue_manager.active_in_zone:
                    rec = self.queue_manager.active_in_zone[track_id]
                    elapsed = (time.time() - rec["entry_time"].timestamp())
                    time_str = format_duration(elapsed)
                    tag_id = f"#{track_id}"
                    tag_time = f"{time_str}"
                else:
                    tag_id = f"#{track_id}"
                    tag_time = ""

                # Compute pill banner dimensions
                (id_w, id_h), _ = cv2.getTextSize(tag_id, font, font_scale, font_thick)
                pad = 6
                if tag_time:
                    (tm_w, tm_h), _ = cv2.getTextSize(tag_time, font, font_scale, font_thick)
                    banner_h = max(id_h, tm_h) + 10
                    banner_w = id_w + tm_w + (pad * 3) + 4
                else:
                    banner_h = id_h + 10
                    banner_w = id_w + (pad * 2)

                # Anchor banner right above bounding box (or inside if near top)
                if y1 - banner_h >= 0:
                    by1 = y1 - banner_h
                    by2 = y1
                else:
                    by1 = y1
                    by2 = y1 + banner_h

                bx1 = x1
                bx2 = x1 + banner_w

                # Draw solid banner background in track's unique color
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, -1)

                # Render text in crisp black
                text_y = by1 + banner_h - 7
                cv2.putText(annotated_frame, tag_id, (bx1 + pad, text_y), font, font_scale, (0, 0, 0), font_thick)

                if tag_time:
                    divider_x = bx1 + pad + id_w + pad
                    cv2.line(annotated_frame, (divider_x, by1 + 3), (divider_x, by2 - 3), (0, 0, 0), 1)
                    cv2.putText(annotated_frame, tag_time, (divider_x + pad, text_y), font, font_scale, (0, 0, 0), font_thick)

            # 6. Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
                self.fps_ready.emit(round(current_fps, 1))

            # 7. Convert to QImage and Emit (Fast multi-threaded pre-scale if large)
            disp_frame = annotated_frame
            if disp_frame.shape[1] > 1280:
                new_w = 1280
                new_h = int(disp_frame.shape[0] * (1280 / disp_frame.shape[1]))
                disp_frame = cv2.resize(disp_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            rgb_frame = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_ready.emit(q_image)

            # 8. Frame Pacing (Smooth 1x playback or maximum unthrottled GPU speed)
            elapsed_time = time.time() - start_tick
            realtime_enabled = getattr(config, "REALTIME_PLAYBACK", True)
            if is_video_file and realtime_enabled:
                sleep_time = max(0.001, (1.0 / target_fps) - elapsed_time)
                self.msleep(int(sleep_time * 1000))
            else:
                self.msleep(1)

        cap.release()
        self.stats_ready.emit(self.queue_manager.update([]))
        self.status_message.emit("Video processing stopped.")

    def stop(self):
        """Stops the processing thread."""
        self.running = False
        self.wait(1000)

    def pause(self):
        """Toggles pause/resume state."""
        self.paused = not self.paused
