import os
import random
import sys

import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("火灾检测测试集预处理工具")
        self.resize(600, 400)

        self.input_dir = ""

        # ===== UI组件 =====
        self.dir_label = QLabel("未选择文件夹")
        self.select_btn = QPushButton("选择图片文件夹")

        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(0, 30)
        self.angle_spin.setValue(30)

        self.flip_check = QCheckBox("启用随机翻转")

        self.process_btn = QPushButton("开始处理")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        # ===== 布局 =====
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.select_btn)
        top_layout.addWidget(self.dir_label)

        layout.addLayout(top_layout)

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("最大旋转角度(°)："))
        angle_layout.addWidget(self.angle_spin)

        layout.addLayout(angle_layout)
        layout.addWidget(self.flip_check)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

        # ===== 信号绑定 =====
        self.select_btn.clicked.connect(self.select_directory)
        self.process_btn.clicked.connect(self.process_images)

    # ------------------ 功能函数 ------------------

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if dir_path:
            self.input_dir = dir_path
            self.dir_label.setText(dir_path)
            self.log("已选择文件夹：" + dir_path)

    def process_images(self):
        if not self.input_dir or not os.path.isdir(self.input_dir):
            self.log("❌ 请先选择有效的图片文件夹")
            return

        output_dir = os.path.join(self.input_dir, "processed")
        os.makedirs(output_dir, exist_ok=True)

        max_angle = self.angle_spin.value()
        enable_flip = self.flip_check.isChecked()

        image_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

        if len(image_files) == 0:
            self.log("❌ 文件夹中没有图片")
            return

        self.log(f"开始处理 {len(image_files)} 张图片...")

        for idx, filename in enumerate(image_files):
            img_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                self.log(f"⚠ 无法读取：{filename}")
                continue

            processed_img = self.augment_image(img, max_angle, enable_flip)

            save_name = f"{idx:03d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, processed_img)

        self.log("✅ 处理完成！")
        self.log(f"输出目录：{output_dir}")

    def augment_image(self, img, max_angle, enable_flip):
        h, w = img.shape[:2]

        # 随机旋转
        angle = random.uniform(-max_angle, max_angle)
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 随机翻转
        if enable_flip:
            flip_code = random.choice([-1, 0, 1])
            rotated = cv2.flip(rotated, flip_code)

        return rotated

    def log(self, text):
        self.log_box.append(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
