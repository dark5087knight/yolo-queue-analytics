"""
Wait Trainer Tool.
Matches Figure in Report (Section 13 / Page 43).
Reads exported queue data CSV, calculates average waiting time and peak operational hour,
and updates config.py directly.
"""
import os
import re
import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import pandas as pd
import config
from queue_manager import format_duration

class WaitTrainerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 + DeepSORT | Wait trainer (config)")
        self.resize(650, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e24;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #3b82f6;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #60a5fa;
                padding-top: 15px;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QLineEdit {
                background-color: #2b2b36;
                color: #ffffff;
                border: 1px solid #3f3f4e;
                border-radius: 4px;
                padding: 6px;
            }
            QTableWidget {
                background-color: #18181f;
                color: #e0e0e0;
                gridline-color: #2c2c38;
                border: 1px solid #2c2c38;
            }
            QHeaderView::section {
                background-color: #282834;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #33333e;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

        self.csv_path: Optional[str] = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Instruction Hint
        lbl_hint = QLabel("Match the queue export format: Customer ID, Waiting Time, Hour, Day.")
        lbl_hint.setFont(QFont("Segoe UI", 9, QFont.StyleItalic))
        lbl_hint.setStyleSheet("color: #94a3b8;")
        layout.addWidget(lbl_hint)

        # File selection box
        grp_csv = QGroupBox("Queue data (CSV)")
        grp_csv.setStyleSheet("QGroupBox { color: #818cf8; font-weight: bold; font-size: 13px; }")
        csv_vbox = QVBoxLayout(grp_csv)
        csv_top_row = QHBoxLayout()

        self.txt_path = QLabel("No file selected")
        self.txt_path.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        self.btn_select_csv = QPushButton("📁 Select CSV")
        self.btn_select_csv.setStyleSheet("background-color: #334155; color: white; padding: 6px 14px; font-weight: bold;")
        self.btn_select_csv.clicked.connect(self.select_csv)

        csv_top_row.addWidget(self.txt_path)
        csv_top_row.addStretch()
        csv_top_row.addWidget(self.btn_select_csv)
        csv_vbox.addLayout(csv_top_row)

        lbl_rule = QLabel(
            "Peak hour = clock hour (0–23) with the most customers (rows). Only config.py is updated — no extra export files."
        )
        lbl_rule.setStyleSheet("color: #94a3b8; font-size: 11px; margin-top: 6px;")
        csv_vbox.addWidget(lbl_rule)
        layout.addWidget(grp_csv)

        # Action button
        self.btn_train = QPushButton("▶ Start training  apply to config.py")
        self.btn_train.setStyleSheet("background-color: #2563eb; color: white; font-size: 13px; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

        # Results Group
        grp_results = QGroupBox("Results")
        grp_results.setStyleSheet("QGroupBox { color: #818cf8; font-weight: bold; font-size: 13px; }")
        results_layout = QVBoxLayout(grp_results)

        self.lbl_summary = QLabel("Pick a CSV, then Start training. Summary and per-hour counts appear here.")
        self.lbl_summary.setFont(QFont("Segoe UI", 10))
        self.lbl_summary.setStyleSheet("color: #94a3b8; padding: 6px;")
        results_layout.addWidget(self.lbl_summary)

        self.tbl_hourly = QTableWidget()
        self.tbl_hourly.setColumnCount(3)
        self.tbl_hourly.setHorizontalHeaderLabels(["Hour (24h)", "Customer Count", "% of Total"])
        self.tbl_hourly.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.tbl_hourly)

        layout.addWidget(grp_results)

    def select_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Queue Data CSV", config.REPORTS_DIR, "CSV Files (*.csv)"
        )
        if path:
            self.csv_path = path
            self.txt_path.setText(path)

    def start_training(self):
        if not self.csv_path or not os.path.exists(self.csv_path):
            QMessageBox.warning(self, "Missing File", "Please select a valid CSV file first.")
            return

        try:
            df = pd.read_csv(self.csv_path)

            # Resolve wait time column
            wait_col = None
            for col in ["Waiting Time", "Wait (Seconds)", "Wait (s)", "wait_seconds"]:
                if col in df.columns:
                    wait_col = col
                    break
            if not wait_col:
                QMessageBox.critical(self, "Invalid CSV", "CSV must contain a 'Waiting Time' or 'Wait (Seconds)' column.")
                return

            # Resolve hour column
            hour_col = None
            for col in ["Hour", "hour"]:
                if col in df.columns:
                    hour_col = col
                    break
            if not hour_col and "Entry Time" in df.columns:
                df["Hour"] = pd.to_datetime(df["Entry Time"]).dt.hour
                hour_col = "Hour"

            if not hour_col:
                QMessageBox.critical(self, "Invalid CSV", "CSV must contain an 'Hour' or 'Entry Time' column.")
                return

            # Calculate calibration metrics
            df[wait_col] = pd.to_numeric(df[wait_col], errors="coerce").fillna(0.0)
            avg_wait_sec = round(float(df[wait_col].mean()), 2)

            hourly_counts = df[hour_col].value_counts().sort_index()
            peak_hour = int(df[hour_col].mode().iloc[0])

            # Apply updates directly to config.py
            self.update_config_file(avg_wait_sec, peak_hour)

            # Display results
            self.lbl_summary.setText(
                f"<b>Training Complete:</b><br>"
                f"• Processed <b>{len(df)}</b> customer visits<br>"
                f"• Calibrated Average Wait Time: <b>{avg_wait_sec}s</b> ({format_duration(avg_wait_sec)})<br>"
                f"• Identified Peak Traffic Hour: <b>{peak_hour}:00 - {peak_hour}:59</b>"
            )

            # Populate hourly table
            self.tbl_hourly.setRowCount(24)
            total_customers = max(1, len(df))
            for h in range(24):
                cnt = int(hourly_counts.get(h, 0))
                pct = (cnt / total_customers) * 100
                item_h = QTableWidgetItem(f"{h:02d}:00")
                item_c = QTableWidgetItem(str(cnt))
                item_p = QTableWidgetItem(f"{pct:.1f}%")

                if h == peak_hour:
                    item_h.setBackground(Qt.darkGreen)
                    item_c.setBackground(Qt.darkGreen)
                    item_p.setBackground(Qt.darkGreen)

                self.tbl_hourly.setItem(h, 0, item_h)
                self.tbl_hourly.setItem(h, 1, item_c)
                self.tbl_hourly.setItem(h, 2, item_p)

            QMessageBox.information(
                self, "Success",
                f"config.py updated successfully!\n\n"
                f"AVERAGE_WAIT_TIME_SECONDS = {avg_wait_sec}\n"
                f"PEAK_HOUR = {peak_hour}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Failed to train from CSV: {str(e)}")

    def update_config_file(self, avg_wait: float, peak_hour: int):
        """Rewrites config.py with the newly calibrated parameters."""
        config_file_path = os.path.join(os.path.dirname(__file__), "config.py")
        with open(config_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Regex replacements for configuration constants
        content = re.sub(
            r"AVERAGE_WAIT_TIME_SECONDS\s*=\s*[\d\.]+",
            f"AVERAGE_WAIT_TIME_SECONDS = {avg_wait}",
            content
        )
        content = re.sub(
            r"PEAK_HOUR\s*=\s*\d+",
            f"PEAK_HOUR = {peak_hour}",
            content
        )

        with open(config_file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Update in-memory config module
        config.AVERAGE_WAIT_TIME_SECONDS = avg_wait
        config.PEAK_HOUR = peak_hour

def main():
    app = QApplication(sys.argv)
    window = WaitTrainerWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
