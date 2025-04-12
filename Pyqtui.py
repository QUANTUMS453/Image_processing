# Combined_UI_YOLO.py
import PyQt6.QtWidgets as qtw
import PyQt6.QtGui as qtg
import sys
import threading
from Process import Detection  # Assuming Process.py is in the same folder


class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO GUI")
        self.setLayout(qtw.QVBoxLayout())

        self.lbl = qtw.QLabel("Welcome to detect app")
        self.lbl.setFont(qtg.QFont("Helvetica", 25))
        self.layout().addWidget(self.lbl)

        self.history = []
        self.selected_path = ""

        open_btn = qtw.QPushButton("Open File", clicked=self.open_file)
        self.layout().addWidget(open_btn)

        print_btn = qtw.QPushButton("Print History", clicked=self.print_history)
        self.layout().addWidget(print_btn)

        clear_btn = qtw.QPushButton("Clear History", clicked=self.clear_history)
        self.layout().addWidget(clear_btn)

        start_btn = qtw.QPushButton("Start Detection", clicked=self.start_detection)
        self.layout().addWidget(start_btn)

        self.show()

    def clear_history(self):
        self.history.clear()
        self.lbl.setText("History cleared!")

    def print_history(self):
        if not self.history:
            self.lbl.setText("No history yet.")
        else:
            self.lbl.setText("\n".join(self.history))

    def open_file(self):
        file_path, _ = qtw.QFileDialog.getOpenFileName(self, "Select a File")

        if file_path:
            self.selected_path = file_path
            if file_path not in self.history:
                self.history.append(file_path)
            self.lbl.setText("\n".join(self.history))

    def start_detection(self):
        if not self.selected_path:
            self.lbl.setText("No file selected!")
            return

        self.lbl.setText("Running detection...")

        # Run YOLO detection in a separate thread
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        model_name = "yolov10x"  #TODO make this dynamic later
        det = Detection(self.selected_path, model_name)
        det.run()


        
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())
