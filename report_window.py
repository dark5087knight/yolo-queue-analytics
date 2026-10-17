"""
Queue Analytics Report Window.
Displays detailed operational summaries, zone statistics, and customer logs.
Provides Export to CSV and Export to PDF functionality.
"""
import os
from datetime import datetime
from typing import List, Dict, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import config
from queue_manager import QueueManager, format_mm_ss

class ReportWindow(QDialog):
    def __init__(self, queue_manager: QueueManager, parent=None):
        super().__init__(parent)
        self.queue_manager = queue_manager
        self.setWindowTitle("Queue Analytics Report")
        self.resize(800, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e24;
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #33333e;
                background-color: #24242d;
            }
            QTabBar::tab {
                background-color: #18181f;
                color: #a0a0a0;
                padding: 8px 20px;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #24242d;
                color: #3b82f6;
            }
            QGroupBox {
                border: 1px solid #3b82f6;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #3b82f6;
                padding-top: 15px;
            }
            QTableWidget {
                background-color: #18181f;
                color: #e0e0e0;
                gridline-color: #2c2c38;
                border: 1px solid #2c2c38;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #282834;
                color: #ffffff;
                padding: 6px;
                border: 1px solid #33333e;
                font-weight: bold;
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
            QLabel {
                color: #e0e0e0;
            }
        """)

        self.init_ui()
        self.refresh_data()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Header toolbar with buttons
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Refresh Report")
        self.btn_refresh.clicked.connect(self.refresh_data)

        self.btn_export_csv = QPushButton("📥 Export CSV")
        self.btn_export_csv.setStyleSheet("background-color: #059669; font-weight: bold;")
        self.btn_export_csv.clicked.connect(self.export_csv)

        self.btn_export_pdf = QPushButton("📄 Export PDF")
        self.btn_export_pdf.setStyleSheet("background-color: #dc2626; font-weight: bold;")
        self.btn_export_pdf.clicked.connect(self.export_pdf)

        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_export_pdf)
        main_layout.addLayout(btn_layout)

        # Tabs
        self.tabs = QTabWidget()

        # Tab 1: Summary
        self.tab_summary = QWidget()
        summary_layout = QVBoxLayout(self.tab_summary)

        # Metrics Box
        grp_metrics = QGroupBox("Overall Statistics")
        grp_metrics.setStyleSheet("QGroupBox { color: #a78bfa; font-weight: bold; font-size: 13px; border: 1px solid #4338ca; }")
        metrics_grid = QGridLayout(grp_metrics)

        self.lbl_total_served = QLabel("Total Customers Served: 0")
        self.lbl_total_served.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_total_served.setStyleSheet("color: #4ade80;")

        self.lbl_avg_wait = QLabel("Average Wait Time: 00:00")
        self.lbl_avg_wait.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_avg_wait.setStyleSheet("color: #60a5fa;")

        self.lbl_min_wait = QLabel("Minimum Wait: 00:00")
        self.lbl_min_wait.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_min_wait.setStyleSheet("color: #38bdf8;")

        self.lbl_max_wait = QLabel("Maximum Wait: 00:00")
        self.lbl_max_wait.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_max_wait.setStyleSheet("color: #f87171;")

        metrics_grid.addWidget(self.lbl_total_served, 0, 0)
        metrics_grid.addWidget(self.lbl_avg_wait, 0, 1)
        metrics_grid.addWidget(self.lbl_min_wait, 1, 0)
        metrics_grid.addWidget(self.lbl_max_wait, 1, 1)
        summary_layout.addWidget(grp_metrics)

        # Zone Stats Table
        lbl_zone = QLabel("Statistics by Zone")
        lbl_zone.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_zone.setStyleSheet("color: #a78bfa; margin-top: 10px; margin-bottom: 4px;")
        summary_layout.addWidget(lbl_zone)

        self.tbl_zones = QTableWidget()
        self.tbl_zones.setColumnCount(5)
        self.tbl_zones.setHorizontalHeaderLabels(["Zone", "Customers", "Avg Wait", "Min Wait", "Max Wait"])
        self.tbl_zones.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(self.tbl_zones)

        self.tabs.addTab(self.tab_summary, "📊 Summary")

        # Tab 2: Customer Details
        self.tab_details = QWidget()
        details_layout = QVBoxLayout(self.tab_details)

        self.tbl_customers = QTableWidget()
        self.tbl_customers.setColumnCount(7)
        self.tbl_customers.setHorizontalHeaderLabels([
            "Customer ID", "Zone", "Entry Time", "Exit Time", "Wait Time (MM:SS)", "Wait (Seconds)", "Status"
        ])
        self.tbl_customers.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        details_layout.addWidget(self.tbl_customers)

        self.tabs.addTab(self.tab_details, "📋 Customer Detail")
        main_layout.addWidget(self.tabs)

    def refresh_data(self):
        """Refreshes summary statistics and tables."""
        stats = self.queue_manager.get_summary_statistics()

        self.lbl_total_served.setText(f"Total Customers Served: {stats['total_served']}")
        self.lbl_avg_wait.setText(f"Average Wait Time: {stats['avg_wait']}")
        self.lbl_min_wait.setText(f"Minimum Wait: {stats['min_wait']}")
        self.lbl_max_wait.setText(f"Maximum Wait: {stats['max_wait']}")

        # Populate zone table
        zone_stats = stats["zone_stats"]
        self.tbl_zones.setRowCount(len(zone_stats))
        for row, z in enumerate(zone_stats):
            self.tbl_zones.setItem(row, 0, QTableWidgetItem(f"Zone {z['zone']}"))
            self.tbl_zones.setItem(row, 1, QTableWidgetItem(str(z["customers"])))
            self.tbl_zones.setItem(row, 2, QTableWidgetItem(z["avg_wait"]))
            self.tbl_zones.setItem(row, 3, QTableWidgetItem(z["min_wait"]))
            self.tbl_zones.setItem(row, 4, QTableWidgetItem(z["max_wait"]))

        # Populate customer details table
        customers = self.queue_manager.served_customers
        self.tbl_customers.setRowCount(len(customers))
        for row, c in enumerate(customers):
            self.tbl_customers.setItem(row, 0, QTableWidgetItem(str(c["customer_id"])))
            self.tbl_customers.setItem(row, 1, QTableWidgetItem(str(c["zone"])))
            self.tbl_customers.setItem(row, 2, QTableWidgetItem(str(c["entry_time"])))
            self.tbl_customers.setItem(row, 3, QTableWidgetItem(str(c["exit_time"])))
            self.tbl_customers.setItem(row, 4, QTableWidgetItem(str(c["wait_time_str"])))
            self.tbl_customers.setItem(row, 5, QTableWidgetItem(str(c["wait_seconds"])))
            self.tbl_customers.setItem(row, 6, QTableWidgetItem(str(c["status"])))

    def export_csv(self):
        """Exports customer log to CSV compatible with Wait Trainer."""
        customers = self.queue_manager.served_customers
        if not customers:
            QMessageBox.information(self, "Export CSV", "No served customers recorded yet.")
            return

        default_filename = f"queue_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        default_path = os.path.join(config.REPORTS_DIR, default_filename)

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Queue CSV", default_path, "CSV Files (*.csv)"
        )
        if not filepath:
            return

        # Prepare DataFrame with required fields for Wait Trainer & Reports
        rows = []
        for c in customers:
            try:
                entry_dt = datetime.strptime(c["entry_time"], "%Y-%m-%d %H:%M:%S")
                hour = entry_dt.hour
                day = entry_dt.strftime("%A")
            except Exception:
                hour = datetime.now().hour
                day = datetime.now().strftime("%A")

            rows.append({
                "Customer ID": c["customer_id"],
                "Waiting Time": c["wait_seconds"],
                "Hour": hour,
                "Day": day,
                "Zone": c["zone"],
                "Entry Time": c["entry_time"],
                "Exit Time": c["exit_time"],
                "Status": c["status"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        QMessageBox.information(self, "Export Successful", f"Report saved to:\n{filepath}")

    def export_pdf(self):
        """Exports formatted PDF operational report using ReportLab."""
        customers = self.queue_manager.served_customers
        if not customers:
            QMessageBox.information(self, "Export PDF", "No served customers recorded yet.")
            return

        default_filename = f"queue_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        default_path = os.path.join(config.REPORTS_DIR, default_filename)

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Queue PDF", default_path, "PDF Files (*.pdf)"
        )
        if not filepath:
            return

        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            # Title
            title_style = ParagraphStyle(
                'ReportTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor("#1e3a8a"),
                spaceAfter=15,
            )
            elements.append(Paragraph("Queue Analytics & Monitoring Report", title_style))

            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elements.append(Paragraph(f"<b>Generated:</b> {time_str}", styles['Normal']))
            elements.append(Spacer(1, 15))

            # Summary Section
            stats = self.queue_manager.get_summary_statistics()
            summary_data = [
                ["Metric", "Value"],
                ["Total Customers Served", str(stats["total_served"])],
                ["Average Wait Time", stats["avg_wait"]],
                ["Minimum Wait Time", stats["min_wait"]],
                ["Maximum Wait Time", stats["max_wait"]],
            ]
            t_summary = Table(summary_data, colWidths=[200, 200])
            t_summary.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(Paragraph("<b>Summary Statistics</b>", styles['Heading2']))
            elements.append(t_summary)
            elements.append(Spacer(1, 20))

            # Customer Details Table
            elements.append(Paragraph("<b>Customer Details Log</b>", styles['Heading2']))
            cust_table_data = [["ID", "Zone", "Entry Time", "Exit Time", "Wait (MM:SS)", "Wait (s)", "Status"]]
            for c in customers[:50]:  # Cap first 50 for clean printing
                cust_table_data.append([
                    str(c["customer_id"]),
                    str(c["zone"]),
                    c["entry_time"].split()[-1],
                    c["exit_time"].split()[-1],
                    c["wait_time_str"],
                    str(c["wait_seconds"]),
                    c["status"],
                ])

            t_cust = Table(cust_table_data, colWidths=[35, 35, 75, 75, 75, 55, 60])
            t_cust.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            elements.append(t_cust)

            doc.build(elements)
            QMessageBox.information(self, "Export Successful", f"PDF report saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to generate PDF: {str(e)}")
