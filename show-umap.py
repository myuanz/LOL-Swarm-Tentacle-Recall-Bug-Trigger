import sys
import pickle
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UMAP Visualization")
        self.setGeometry(100, 100, 800, 600)

        # Load data
        with open('chrs.pickle', 'rb') as f:
            data = pickle.load(f)
        self.u = data['u']
        self.imgs = data['imgs']

        # Create main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        # Create matplotlib figure and add it to the layout
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Plot the scatter
        self.ax = self.figure.add_subplot(111)
        self.scatter = self.ax.scatter(self.u[:, 0], self.u[:, 1], picker=5, s=20)  # 5 points tolerance
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Create image display widget
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.setCentralWidget(main_widget)

    def on_hover(self, event):
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                i = ind["ind"][0]
                img = self.imgs[i]
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.image_label.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())