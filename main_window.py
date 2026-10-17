"""
Main Window for YOLOv8 + DeepSORT Queue Detection and Analytics.
Implements the exact layout and controls from the project documentation and GUI reference screenshots.
"""
import torch  # Crucial: Must be imported before PyQt5 on Windows
import os
import sys
from typing import List, Tuple, Optional
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QGroupBox, QSpinBox,
    QDoubleSpinBox, QStatusBar, QFrame, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QPoint, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

import config
from queue_manager import QueueManager
from detection_thread import DetectionThread
from report_window import ReportWindow

class InteractiveVideoWidget(QLabel):
    """
    Video display label with interactive polygon and rectangle drawing capabilities.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px;")
        self.setText("Select a video or press the Camera button")
        self.setFont(QFont("Segoe UI", 12))

        self.draw_mode: Optional[str] = None  # 'drag' or '4pts'
        self.current_points: List[Tuple[int, int]] = []
        self.drag_start: Optional[QPoint] = None
        self.drag_current: Optional[QPoint] = None

        self.zone_callback = None
        self.current_pixmap: Optional[QPixmap] = None

    def set_frame(self, q_image: QImage):
        """Displays the updated video frame."""
        pixmap = QPixmap.fromImage(q_image)
        self.current_pixmap = pixmap
        # Fast GPU/SIMD-friendly scaling to widget size
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if not self.draw_mode:
            return

        pos = event.pos()
        # Map label coordinates to image coordinates
        img_pt = self.map_to_image_coords(pos)
        if not img_pt:
            return

        if self.draw_mode == "drag":
            self.drag_start = pos
            self.drag_current = pos
            self.update()

        elif self.draw_mode == "4pts":
            self.current_points.append(img_pt)
            self.update()
            if len(self.current_points) == 4:
                if self.zone_callback:
                    self.zone_callback(self.current_points.copy())
                self.current_points.clear()
                self.draw_mode = None
                self.update()

    def mouseMoveEvent(self, event):
        if self.draw_mode == "drag" and self.drag_start is not None:
            self.drag_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.draw_mode == "drag" and self.drag_start is not None:
            p1 = self.map_to_image_coords(self.drag_start)
            p2 = self.map_to_image_coords(event.pos())
            self.drag_start = None
            self.drag_current = None
            self.draw_mode = None
            self.update()

            if p1 and p2:
                x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
                y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
                rect_pts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                if self.zone_callback:
                    self.zone_callback(rect_pts)

    def map_to_image_coords(self, pt: QPoint) -> Optional[Tuple[int, int]]:
        """Maps click on QLabel to the original video frame pixel coordinates."""
        if not self.current_pixmap:
            # Fallback to direct label coordinates
            return (pt.x(), pt.y())

        lbl_w = self.width()
        lbl_h = self.height()
        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()

        # Compute scaling and letterbox offsets
        scale = min(lbl_w / img_w, lbl_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        off_x = (lbl_w - disp_w) // 2
        off_y = (lbl_h - disp_h) // 2

        if off_x <= pt.x() <= off_x + disp_w and off_y <= pt.y() <= off_y + disp_h:
            orig_x = int((pt.x() - off_x) / scale)
            orig_y = int((pt.y() - off_y) / scale)
            return (orig_x, orig_y)

        return None

    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw in-progress shapes on top of the pixmap
        if self.draw_mode == "drag" and self.drag_start and self.drag_current:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 215, 0), 2, Qt.DashLine))
            painter.drawRect(
                min(self.drag_start.x(), self.drag_current.x()),
                min(self.drag_start.y(), self.drag_current.y()),
                abs(self.drag_current.x() - self.drag_start.x()),
                abs(self.drag_current.y() - self.drag_start.y()),
            )
        elif self.draw_mode == "4pts" and self.current_points:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            painter.setBrush(QColor(0, 255, 255, 120))
            for pt in self.current_points:
                painter.drawEllipse(QPoint(pt[0], pt[1]), 4, 4)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 + DeepSORT | Queue Detection & Tracking")
        self.resize(1200, 750)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QWidget {
                color: #f8fafc;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                background-color: #1e293b;
            }
            QTabBar::tab {
                background-color: #0f172a;
                color: #94a3b8;
                padding: 8px 18px;
                font-weight: bold;
                border: 1px solid #334155;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 4px;
                padding: 7px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #475569;
                color: #94a3b8;
            }
            QTableWidget {
                background-color: #0f172a;
                color: #f8fafc;
                gridline-color: #334155;
                border: 1px solid #334155;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #334155;
            }
            QGroupBox {
                border: 1px solid #38bdf8;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #38bdf8;
                padding-top: 12px;
            }
            QStatusBar {
                background-color: #0b1120;
                color: #94a3b8;
                font-weight: bold;
            }
        """)

        self.queue_manager = QueueManager()
        self.detection_thread: Optional[DetectionThread] = None
        self.current_source: Optional[str] = None
        self.report_dialog: Optional[ReportWindow] = None
        self.last_fps: float = 0.0
        self.last_stats: dict = {}
        self.preview_frame = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Upper Body: Left Video + Right Sidebar
        body_layout = QHBoxLayout()

        # Left: Interactive Video View
        self.video_view = InteractiveVideoWidget()
        self.video_view.zone_callback = self.on_zone_defined
        body_layout.addWidget(self.video_view, stretch=3)

        # Right: Tabs (Settings / Queue)
        self.tabs = QTabWidget()

        # --- Queue Tab ---
        tab_queue = QWidget()
        queue_layout = QVBoxLayout(tab_queue)

        # Live Summary Card
        grp_summary = QGroupBox("Live Summary")
        summary_layout = QVBoxLayout(grp_summary)

        self.lbl_in_queue = QLabel("In Queue: 0")
        self.lbl_in_queue.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_in_queue.setStyleSheet("color: #38bdf8; margin-bottom: 2px;")

        self.lbl_entering = QLabel("Entering: 0")
        self.lbl_entering.setFont(QFont("Segoe UI", 10))
        self.lbl_entering.setStyleSheet("color: #94a3b8; margin-bottom: 2px;")

        self.lbl_served = QLabel("Served: 0")
        self.lbl_served.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_served.setStyleSheet("color: #4ade80; margin-bottom: 4px;")

        summary_layout.addWidget(self.lbl_in_queue)
        summary_layout.addWidget(self.lbl_entering)
        summary_layout.addWidget(self.lbl_served)
        queue_layout.addWidget(grp_summary)

        # Currently in Queue Table
        lbl_tbl_q = QLabel("Currently in Queue:")
        lbl_tbl_q.setFont(QFont("Segoe UI", 10, QFont.Bold))
        queue_layout.addWidget(lbl_tbl_q)

        self.tbl_queue = QTableWidget()
        self.tbl_queue.setColumnCount(4)
        self.tbl_queue.setHorizontalHeaderLabels(["ID", "Zone", "Waited", "Est. wait"])
        self.tbl_queue.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        queue_layout.addWidget(self.tbl_queue)

        # Recently Served (last 10) Table
        lbl_tbl_s = QLabel("Recently Served (last 10):")
        lbl_tbl_s.setFont(QFont("Segoe UI", 10, QFont.Bold))
        queue_layout.addWidget(lbl_tbl_s)

        self.tbl_served = QTableWidget()
        self.tbl_served.setColumnCount(3)
        self.tbl_served.setHorizontalHeaderLabels(["ID", "Zone", "Total Wait"])
        self.tbl_served.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        queue_layout.addWidget(self.tbl_served)

        # Queue Tab Action Buttons
        btn_box = QHBoxLayout()
        self.btn_clear_record = QPushButton("Clear record")
        self.btn_clear_record.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.btn_clear_record.clicked.connect(self.clear_records)

        self.btn_full_report = QPushButton("📊 Generate Full Report")
        self.btn_full_report.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.btn_full_report.clicked.connect(self.open_full_report)

        btn_box.addWidget(self.btn_clear_record)
        btn_box.addWidget(self.btn_full_report)
        queue_layout.addLayout(btn_box)

        self.tabs.addTab(tab_queue, "Queue")

        # --- Settings Tab ---
        tab_settings = QWidget()
        settings_layout = QVBoxLayout(tab_settings)

        # Model Selector
        grp_model = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(grp_model)

        self.lbl_current_model = QLabel(f"Model: {os.path.basename(config.DEFAULT_MODEL_PATH)}")
        self.lbl_current_model.setWordWrap(True)
        self.btn_choose_model = QPushButton("Select Custom Model (.pt)")
        self.btn_choose_model.clicked.connect(self.choose_model)

        model_layout.addWidget(self.lbl_current_model)
        model_layout.addWidget(self.btn_choose_model)
        settings_layout.addWidget(grp_model)

        # Parameters Box
        grp_params = QGroupBox("Queue Parameters")
        params_layout = QVBoxLayout(grp_params)

        lbl_conf = QLabel("Confidence Threshold:")
        self.spn_conf = QDoubleSpinBox()
        self.spn_conf.setRange(0.1, 0.95)
        self.spn_conf.setSingleStep(0.05)
        self.spn_conf.setValue(config.CONFIDENCE_THRESHOLD)

        lbl_alert_thresh = QLabel("Alert Overload Threshold:")
        self.spn_alert = QSpinBox()
        self.spn_alert.setRange(1, 50)
        self.spn_alert.setValue(config.ALERT_QUEUE_THRESHOLD)
        self.spn_alert.valueChanged.connect(self.update_alert_thresh)

        params_layout.addWidget(lbl_conf)
        params_layout.addWidget(self.spn_conf)
        params_layout.addWidget(lbl_alert_thresh)
        params_layout.addWidget(self.spn_alert)
        settings_layout.addWidget(grp_params)

        # Multi-Object Tracking Settings Box
        grp_tracking = QGroupBox("Object Tracking Configuration")
        tracking_layout = QVBoxLayout(grp_tracking)

        lbl_engine = QLabel("Tracking Engine:")
        self.cmb_tracker = QComboBox()
        self.cmb_tracker.addItems(["Deep SORT (Appearance + Kalman)", "ByteTrack (YOLO High-Speed)"])
        self.cmb_tracker.setCurrentIndex(0 if getattr(config, "TRACKER_TYPE", "deepsort") == "deepsort" else 1)
        self.cmb_tracker.currentIndexChanged.connect(self.update_tracker_type)

        self.chk_trajectories = QCheckBox("Show Motion Trajectory Trails")
        self.chk_trajectories.setChecked(getattr(config, "SHOW_TRAJECTORIES", True))
        self.chk_trajectories.stateChanged.connect(self.update_trajectory_display)

        lbl_trail_len = QLabel("Trajectory History Length:")
        self.spn_trail_len = QSpinBox()
        self.spn_trail_len.setRange(5, 120)
        self.spn_trail_len.setValue(getattr(config, "TRAJECTORY_LENGTH", 30))
        self.spn_trail_len.valueChanged.connect(self.update_trajectory_length)

        self.chk_realtime = QCheckBox("Real-Time Frame Pacing (1x Video Speed)")
        self.chk_realtime.setChecked(getattr(config, "REALTIME_PLAYBACK", True))
        self.chk_realtime.setToolTip("Uncheck to process at maximum GPU speed without artificial sleep.")
        self.chk_realtime.stateChanged.connect(self.update_realtime_playback)

        tracking_layout.addWidget(lbl_engine)
        tracking_layout.addWidget(self.cmb_tracker)
        tracking_layout.addWidget(self.chk_trajectories)
        tracking_layout.addWidget(lbl_trail_len)
        tracking_layout.addWidget(self.spn_trail_len)
        tracking_layout.addWidget(self.chk_realtime)
        settings_layout.addWidget(grp_tracking)

        # Operational Insights Box
        grp_info = QGroupBox("Calibrated Operational Values")
        info_layout = QVBoxLayout(grp_info)
        self.lbl_calib_wait = QLabel(f"Average Service Wait: {config.AVERAGE_WAIT_TIME_SECONDS}s")
        self.lbl_calib_peak = QLabel(f"Peak Traffic Hour: {config.PEAK_HOUR}:00 - {config.PEAK_HOUR}:59")
        info_layout.addWidget(self.lbl_calib_wait)
        info_layout.addWidget(self.lbl_calib_peak)
        settings_layout.addWidget(grp_info)

        # Hardware & GPU Status Box
        grp_hw = QGroupBox("Hardware & GPU Acceleration")
        hw_layout = QVBoxLayout(grp_hw)
        if getattr(config, "USE_CUDA", False):
            lbl_gpu_dev = QLabel(f"Device: {config.GPU_NAME}")
            lbl_gpu_dev.setStyleSheet("color: #4ade80; font-weight: bold;")
            lbl_gpu_vram = QLabel(f"GPU VRAM: {config.GPU_VRAM_GB:.1f} GB Dedicated")
            lbl_gpu_cuda = QLabel("CUDA Acceleration: Active (CUDA / PyTorch)")
            lbl_gpu_cuda.setStyleSheet("color: #38bdf8;")
            hw_layout.addWidget(lbl_gpu_dev)
            hw_layout.addWidget(lbl_gpu_vram)
            hw_layout.addWidget(lbl_gpu_cuda)
        else:
            lbl_cpu = QLabel("Running on CPU (CUDA not detected)")
            lbl_cpu.setStyleSheet("color: #facc15;")
            hw_layout.addWidget(lbl_cpu)
        settings_layout.addWidget(grp_hw)

        settings_layout.addStretch()
        self.tabs.addTab(tab_settings, "Settings")

        body_layout.addWidget(self.tabs, stretch=1)
        main_layout.addLayout(body_layout)

        # Bottom Control Toolbar
        toolbar_layout = QHBoxLayout()

        self.btn_select_video = QPushButton("📁 Select Video")
        self.btn_select_video.clicked.connect(self.select_video)

        self.btn_camera = QPushButton("📷 Camera")
        self.btn_camera.clicked.connect(self.select_camera)

        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setStyleSheet("background-color: #16a34a;")
        self.btn_start.clicked.connect(self.start_processing)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setStyleSheet("background-color: #dc2626;")
        self.btn_stop.clicked.connect(self.stop_processing)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setStyleSheet("background-color: #0284c7;")
        self.btn_pause.clicked.connect(self.pause_processing)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #475569;")

        self.btn_zone_drag = QPushButton("✏ Zone (Drag)")
        self.btn_zone_drag.clicked.connect(self.enable_zone_drag)

        self.btn_zone_4pts = QPushButton("✏ Zone (4 Pts)")
        self.btn_zone_4pts.clicked.connect(self.enable_zone_4pts)

        self.btn_undo_zone = QPushButton("↶ Undo Zone")
        self.btn_undo_zone.setStyleSheet("background-color: #475569;")
        self.btn_undo_zone.clicked.connect(self.undo_zone)

        self.btn_clear_zones = QPushButton("🗑 Clear Zones")
        self.btn_clear_zones.setStyleSheet("background-color: #475569;")
        self.btn_clear_zones.clicked.connect(self.clear_zones)

        self.lbl_toolbar_zones = QLabel("Zones: 0")
        self.lbl_toolbar_zones.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 12px; margin-left: 6px;")

        toolbar_layout.addWidget(self.btn_select_video)
        toolbar_layout.addWidget(self.btn_camera)
        toolbar_layout.addWidget(self.btn_start)
        toolbar_layout.addWidget(self.btn_stop)
        toolbar_layout.addWidget(self.btn_pause)
        toolbar_layout.addWidget(sep)
        toolbar_layout.addWidget(self.btn_zone_drag)
        toolbar_layout.addWidget(self.btn_zone_4pts)
        toolbar_layout.addWidget(self.btn_undo_zone)
        toolbar_layout.addWidget(self.btn_clear_zones)
        toolbar_layout.addWidget(self.lbl_toolbar_zones)
        toolbar_layout.addStretch()

        main_layout.addLayout(toolbar_layout)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_status_metrics = QLabel("In queue: 0 | Served: 0 | FPS: --")
        self.lbl_status_metrics.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px; margin-right: 12px;")

        self.status_badge = QLabel("Status: Normal")
        self.status_badge.setStyleSheet(
            "background-color: #16a34a; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px;"
        )

        # GPU Status Badge
        if getattr(config, "USE_CUDA", False):
            gpu_display = config.GPU_NAME.replace("NVIDIA GeForce ", "")
            self.lbl_gpu_badge = QLabel(f"⚡ GPU: {gpu_display}")
            self.lbl_gpu_badge.setStyleSheet(
                "background-color: #1e1b4b; color: #a78bfa; border: 1px solid #6366f1; "
                "padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px; margin-right: 8px;"
            )
        else:
            self.lbl_gpu_badge = QLabel("CPU Mode")
            self.lbl_gpu_badge.setStyleSheet(
                "background-color: #334155; color: #94a3b8; padding: 2px 8px; border-radius: 3px; font-size: 11px; margin-right: 8px;"
            )

        self.status_bar.addPermanentWidget(self.lbl_gpu_badge)
        self.status_bar.addPermanentWidget(self.lbl_status_metrics)
        self.status_bar.addPermanentWidget(self.status_badge)

    def render_preview(self):
        """Renders the preview frame with defined queue zones while the detector is idle."""
        if self.preview_frame is None:
            return
        frame_copy = self.preview_frame.copy()
        for zone in self.queue_manager.zones:
            pts = zone.polygon.reshape((-1, 1, 2))
            cv2.polylines(frame_copy, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            if len(zone.points) > 0:
                lx, ly = zone.points[0]
                cv2.putText(
                    frame_copy, f"Zone {zone.zone_id}", (lx, max(22, ly - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.video_view.set_frame(q_img)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.current_source = path
            fname = os.path.basename(path)
            self.btn_select_video.setText(f"📁 {fname}")
            
            # Load and display the first frame
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.preview_frame = frame
                    self.render_preview()
                cap.release()
            else:
                self.video_view.setText(f"Loaded: {fname}\nPress Start to begin.")
            self.status_bar.showMessage(f"Selected video: {path}", 4000)

    def select_camera(self):
        self.current_source = "0"
        self.btn_select_video.setText("📁 Select Video")
        self.preview_frame = None
        self.video_view.setText("Live Camera Source (Index 0) selected.\nPress Start to begin.")
        self.status_bar.showMessage("Selected Camera 0", 4000)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", config.MODELS_DIR, "PyTorch Model (*.pt)"
        )
        if path:
            config.DEFAULT_MODEL_PATH = path
            self.lbl_current_model.setText(f"Model: {os.path.basename(path)}")
            QMessageBox.information(self, "Model Selected", f"Loaded model weights:\n{path}")

    def update_alert_thresh(self, val):
        config.ALERT_QUEUE_THRESHOLD = val

    def update_tracker_type(self, idx):
        config.TRACKER_TYPE = "deepsort" if idx == 0 else "bytetrack"
        engine_name = "Deep SORT" if idx == 0 else "ByteTrack"
        self.status_bar.showMessage(f"Active Object Tracker changed to {engine_name}.", 4000)

    def update_trajectory_display(self, state):
        config.SHOW_TRAJECTORIES = (state == Qt.Checked)

    def update_trajectory_length(self, val):
        config.TRAJECTORY_LENGTH = val

    def update_realtime_playback(self, state):
        config.REALTIME_PLAYBACK = (state == Qt.Checked)
        mode = "1x Real-Time Video FPS" if config.REALTIME_PLAYBACK else "Maximum Unthrottled GPU Speed"
        self.status_bar.showMessage(f"Playback pacing mode: {mode}", 4000)

    def enable_zone_drag(self):
        self.video_view.draw_mode = "drag"
        self.status_bar.showMessage("Click and drag on the video to define a rectangular queue zone.", 5000)

    def enable_zone_4pts(self):
        self.video_view.draw_mode = "4pts"
        self.video_view.current_points.clear()
        self.status_bar.showMessage("Click 4 points on the video to define a polygon queue zone.", 5000)

    def on_zone_defined(self, points: List[Tuple[int, int]]):
        zone_id = self.queue_manager.add_zone(points)
        self.lbl_toolbar_zones.setText(f"Zones: {len(self.queue_manager.zones)}")
        self.status_bar.showMessage(f"Defined Queue Zone {zone_id} with {len(points)} points.", 4000)
        if self.detection_thread is None or not self.detection_thread.isRunning():
            self.render_preview()

    def undo_zone(self):
        if self.queue_manager.undo_zone():
            self.lbl_toolbar_zones.setText(f"Zones: {len(self.queue_manager.zones)}")
            self.status_bar.showMessage("Last zone removed.", 3000)
            if self.detection_thread is None or not self.detection_thread.isRunning():
                self.render_preview()

    def clear_zones(self):
        self.queue_manager.clear_zones()
        if self.detection_thread:
            self.detection_thread.track_history.clear()
            self.detection_thread.tracker.reset()
        self.lbl_toolbar_zones.setText("Zones: 0")
        self.status_bar.showMessage("All zones cleared.", 3000)
        if self.detection_thread is None or not self.detection_thread.isRunning():
            self.render_preview()

    def clear_records(self):
        self.queue_manager.clear_records()
        self.tbl_queue.setRowCount(0)
        self.tbl_served.setRowCount(0)
        self.lbl_in_queue.setText("In Queue: 0")
        self.lbl_entering.setText("Entering: 0")
        self.lbl_served.setText("Served: 0")
        self.lbl_status_metrics.setText("In queue: 0 | Served: 0 | FPS: --")
        self.status_badge.setText("Status: Normal")
        self.status_badge.setStyleSheet("background-color: #16a34a; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px;")
        self.status_bar.showMessage("Active queue and served records reset.", 3000)

    def start_processing(self):
        if not self.current_source:
            QMessageBox.warning(self, "No Source", "Please select a video file or camera first.")
            return

        if self.detection_thread and self.detection_thread.isRunning():
            return

        self.detection_thread = DetectionThread(
            source=self.current_source,
            queue_manager=self.queue_manager,
            model_path=config.DEFAULT_MODEL_PATH,
            conf_threshold=self.spn_conf.value(),
        )

        self.detection_thread.frame_ready.connect(self.video_view.set_frame)
        self.detection_thread.stats_ready.connect(self.update_dashboard_stats)
        self.detection_thread.fps_ready.connect(self.update_fps)
        self.detection_thread.status_message.connect(self.status_bar.showMessage)
        self.detection_thread.error_occurred.connect(lambda err: QMessageBox.critical(self, "Error", err))

        self.detection_thread.start()
        self.status_bar.showMessage("Queue detection started.", 3000)

    def stop_processing(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread = None
            self.status_bar.showMessage("Queue detection stopped.", 3000)

    def pause_processing(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.pause()
            txt = "Resumed" if not self.detection_thread.paused else "Paused"
            self.status_bar.showMessage(f"Processing {txt}.", 2000)

    @pyqtSlot(dict)
    def update_dashboard_stats(self, stats: dict):
        self.last_stats = stats
        in_q = stats["in_queue"]
        ent = stats["entering"]
        srv = stats["served"]
        status = stats["status"]
        zone_counts = stats.get("zone_counts", {})

        self.lbl_in_queue.setText(f"In Queue: {in_q}")
        self.lbl_entering.setText(f"Entering: {ent}")
        self.lbl_served.setText(f"Served: {srv}")

        # Update Status Badge
        if status == "ALERT":
            self.status_badge.setText("Status: ALERT")
            self.status_badge.setStyleSheet("background-color: #dc2626; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px;")
        elif status == "Busy":
            self.status_badge.setText("Status: Busy")
            self.status_badge.setStyleSheet("background-color: #d97706; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px;")
        else:
            self.status_badge.setText("Status: Normal")
            self.status_badge.setStyleSheet("background-color: #16a34a; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px;")

        # Update Status Bar Metric String: "In queue: 2 | Served: 1 | FPS: 11.5 | Z1: 2"
        fps_str = f"{self.last_fps:.1f}" if self.last_fps > 0 else "--"
        z_str = ""
        if zone_counts:
            z_str = " | " + " | ".join(f"Z{z}: {c}" for z, c in sorted(zone_counts.items()))
        self.lbl_status_metrics.setText(f"In queue: {in_q} | Served: {srv} | FPS: {fps_str}{z_str}")

        # Update Currently in Queue Table
        q_rows = stats["queue_table"]
        self.tbl_queue.setRowCount(len(q_rows))
        for r, row in enumerate(q_rows):
            item_id = QTableWidgetItem(row["id"])
            item_zone = QTableWidgetItem(row["zone"])
            item_waited = QTableWidgetItem(row["waited"])
            item_waited.setForeground(QColor(56, 189, 248))  # Cyan #38bdf8 matching gui_04
            item_est = QTableWidgetItem(row["est_wait"])
            item_est.setForeground(QColor(251, 191, 36))    # Amber #fbbf24 matching gui_04

            self.tbl_queue.setItem(r, 0, item_id)
            self.tbl_queue.setItem(r, 1, item_zone)
            self.tbl_queue.setItem(r, 2, item_waited)
            self.tbl_queue.setItem(r, 3, item_est)

        # Update Recently Served Table
        s_rows = stats["served_table"]
        self.tbl_served.setRowCount(len(s_rows))
        for r, row in enumerate(s_rows):
            self.tbl_served.setItem(r, 0, QTableWidgetItem(row["id"]))
            self.tbl_served.setItem(r, 1, QTableWidgetItem(row["zone"]))
            item_wait = QTableWidgetItem(row["total_wait"])
            item_wait.setForeground(QColor(74, 222, 128))  # Bright Green #4ade80 matching gui_04
            self.tbl_served.setItem(r, 2, item_wait)

    @pyqtSlot(float)
    def update_fps(self, fps: float):
        self.last_fps = fps
        in_q = self.last_stats.get("in_queue", 0)
        srv = self.last_stats.get("served", 0)
        zone_counts = self.last_stats.get("zone_counts", {})
        z_str = ""
        if zone_counts:
            z_str = " | " + " | ".join(f"Z{z}: {c}" for z, c in sorted(zone_counts.items()))
        self.lbl_status_metrics.setText(f"In queue: {in_q} | Served: {srv} | FPS: {fps:.1f}{z_str}")

    def open_full_report(self):
        self.report_dialog = ReportWindow(self.queue_manager, self)
        self.report_dialog.show()

    def closeEvent(self, event):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
