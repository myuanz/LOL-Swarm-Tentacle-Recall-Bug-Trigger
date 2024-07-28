import sys
import os
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtCore import Qt
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QMessageBox

class ImageLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling App")
        self.setGeometry(100, 100, 800, 600)

        self.image_folder = Path("./data/2024-07-25-21-36-48")
        self.labels_file = self.image_folder / "labels.json"
        self.preds_file = self.image_folder / "preds.npy"
        if self.preds_file.exists():
            self.preds = np.load(self.preds_file)
        else:
            self.preds = None

        self.label_types = ["普通人物", "动作人物", "选择卡片", "其他"]
        self.image_files = [f for f in sorted(os.listdir(self.image_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if self.preds is not None and len(self.preds) > 0:
            assert len(self.image_files) == len(self.preds), self.preds

        self.current_index = 0

        self.load_labels()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        label_button_layout = QHBoxLayout()
        self.buttons = []
        for i, label_type in enumerate(self.label_types):
            button = QPushButton(label_type)
            button.clicked.connect(lambda checked, lt=label_type: self.label_image(lt))
            label_button_layout.addWidget(button)
            self.buttons.append(button)
            
        utils_button_layout = QHBoxLayout()
        next_action_button = QPushButton("Next Action")
        next_action_button.clicked.connect(self.to_next_action)
        save_button = QPushButton("Save Labels")
        save_button.clicked.connect(self.save_labels)
        load_pred_button = QPushButton("Load Preds")
        load_pred_button.clicked.connect(self.load_preds)

        utils_button_layout.addWidget(next_action_button)
        utils_button_layout.addWidget(save_button)
        utils_button_layout.addWidget(load_pred_button)

        main_layout.addLayout(label_button_layout)
        main_layout.addLayout(utils_button_layout)

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

    def load_preds(self):
        if self.preds is None:
            return
        if len(self.labels) != 0:
            msg_box = QMessageBox()
            msg_box.setText("真的要载入吗? 会覆盖已有标签")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            response = msg_box.exec()

            if response != QMessageBox.Yes:
                return

        print("Loading preds...", self.image_files)
        for pred, image_file in zip(self.preds, self.image_files):
            self.labels[image_file] = self.label_types[pred.argmax()]
        QMessageBox.information(self, "Success", f"载入成功 {len(self.labels)} 个标签")
        
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
        # self.save_labels()
        self.show_next()

    def to_next_action(self):
        while self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            if self.labels[self.image_files[self.current_index]] == "动作人物":
                self.show_image()
                break

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