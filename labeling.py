import sys
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtCore import Qt

class ImageLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling App")
        self.setGeometry(100, 100, 800, 600)

        self.image_folder = "./data/frames"
        self.labels_file = "./labels.json"
        self.label_types = ["普通人物", "动作人物", "选择卡片", "其他"]
        self.image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0

        self.load_labels()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        self.buttons = []
        for i, label_type in enumerate(self.label_types):
            button = QPushButton(label_type)
            button.clicked.connect(lambda checked, lt=label_type: self.label_image(lt))
            button_layout.addWidget(button)
            self.buttons.append(button)

        main_layout.addLayout(button_layout)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)

        main_layout.addLayout(nav_layout)

        self.status_label = QLabel()
        main_layout.addWidget(self.status_label)

        self.show_image()

    def load_labels(self):
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

        # Find the first unlabeled image
        for i, image_file in enumerate(self.image_files):
            if image_file not in self.labels:
                self.current_index = i
                break

    def show_image(self):
        if 0 <= self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_folder, self.image_files[self.current_index])
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.update_status()
            self.update_button_styles()

    def label_image(self, label_type):
        current_image = self.image_files[self.current_index]
        self.labels[current_image] = label_type
        self.save_labels()
        self.show_next()

    def show_next(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def save_labels(self):
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)

    def update_status(self):
        current_image = self.image_files[self.current_index]
        status = f"Image {self.current_index + 1} of {len(self.image_files)}: {current_image}"
        if current_image in self.labels:
            status += f" (Labeled: {self.labels[current_image]})"
        self.status_label.setText(status)

    def update_button_styles(self):
        current_image = self.image_files[self.current_index]
        current_label = self.labels.get(current_image)
        
        for button, label_type in zip(self.buttons, self.label_types):
            if label_type == current_label:
                button.setStyleSheet("background-color: lightgreen;")
            else:
                button.setStyleSheet("")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() >= Qt.Key_1 and event.key() <= Qt.Key_4:
            index = event.key() - Qt.Key_1
            if index < len(self.label_types):
                self.label_image(self.label_types[index])
        elif event.key() == Qt.Key_A:
            self.show_previous()
        elif event.key() == Qt.Key_D:
            self.show_next()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLabelingApp()
    window.show()
    sys.exit(app.exec())