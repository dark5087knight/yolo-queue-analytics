"""
Main application launcher for the YOLO Queue Detection & Tracking System.
"""
import torch  # Crucial: Must be imported before PyQt5 on Windows
import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

def main():
    print("=" * 65)
    print("  YOLOv8 + DeepSORT Queue Detection & Tracking System")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [⚡ GPU Acceleration: ENABLED]")
        print(f"  • Device: {gpu_name}")
        print(f"  • VRAM:   {vram:.1f} GB")
        print(f"  • Backend: PyTorch {torch.__version__} (CUDA)")
    else:
        print("  [!] CUDA GPU not detected. Running on CPU.")
    print("=" * 65)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
