import json
import os
import sys
import time

import cv2
import ultralytics.nn.tasks as _nn_tasks
from ultralytics import YOLO

# 注册 CBAM 自定义模块到 ultralytics 命名空间
from cbam import CBAM, ChannelAttention, SpatialAttention

_nn_tasks.CBAM = CBAM
_nn_tasks.ChannelAttention = ChannelAttention
_nn_tasks.SpatialAttention = SpatialAttention

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ✅ 导入你提供的邮件发送函数
from message import mailToMeWithImage


class EmailSettingsDialog(QDialog):
    """邮件配置对话框：填写发件人、收件人和授权码."""

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("邮件设置")
        self.setFixedWidth(420)

        layout = QVBoxLayout(self)

        group = QGroupBox("163 邮件告警配置")
        form = QFormLayout(group)
        form.setSpacing(10)

        self.edit_sender = QLineEdit(cfg.get("sender", ""))
        self.edit_sender.setPlaceholderText("例：yourname@163.com")
        form.addRow("发件人（163邮箱）：", self.edit_sender)

        self.edit_receiver = QLineEdit(cfg.get("receiver", ""))
        self.edit_receiver.setPlaceholderText("例：someone@qq.com 或手机邮箱")
        form.addRow("收件人邮箱：", self.edit_receiver)

        self.edit_password = QLineEdit(cfg.get("password", ""))
        self.edit_password.setPlaceholderText("163邮箱客户端授权码，非登录密码")
        self.edit_password.setEchoMode(QLineEdit.Password)
        form.addRow("163客户端授权码：", self.edit_password)

        layout.addWidget(group)

        self.btn_test = QPushButton("测试发送（验证配置是否正确）")
        self.btn_test.clicked.connect(self._on_test_send)
        layout.addWidget(self.btn_test)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("保存")
        btn_box.button(QDialogButtonBox.Cancel).setText("取消")
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self):
        if (
            not self.edit_sender.text().strip()
            or not self.edit_receiver.text().strip()
            or not self.edit_password.text().strip()
        ):
            QMessageBox.warning(self, "信息不完整", "请填写全部三项信息后再保存！")
            return
        self.accept()

    def _on_test_send(self):
        sender = self.edit_sender.text().strip()
        receiver = self.edit_receiver.text().strip()
        password = self.edit_password.text().strip()
        if not sender or not receiver or not password:
            QMessageBox.warning(self, "信息不完整", "请先填写全部三项信息再测试！")
            return
        self.btn_test.setEnabled(False)
        self.btn_test.setText("发送中...")
        try:
            import numpy as np

            dummy_img = np.zeros((100, 300, 3), dtype=np.uint8)
            mailToMeWithImage(
                "测试邮件", "这是一封来自火灾检测系统的测试邮件，配置正常。", dummy_img, sender, receiver, password
            )
            QMessageBox.information(self, "测试成功", "测试邮件已发送，请检查收件箱！")
        except Exception as e:
            QMessageBox.critical(self, "测试失败", f"发送失败，请检查配置：\n{e}")
        finally:
            self.btn_test.setEnabled(True)
            self.btn_test.setText("测试发送（验证配置是否正确）")

    def get_config(self) -> dict:
        return {
            "sender": self.edit_sender.text().strip(),
            "receiver": self.edit_receiver.text().strip(),
            "password": self.edit_password.text().strip(),
        }


class YOLOv8DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 实时目标检测系统（带邮件告警）")
        self.setGeometry(100, 100, 1200, 700)

        # ======================
        # 模型与路径初始化
        # ======================
        # 兼容 PyInstaller 打包：frozen 模式下从临时解压目录找模型，否则从脚本目录找
        if getattr(sys, "frozen", False):
            _base = sys._MEIPASS
        else:
            _base = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(_base, "best.pt")
        self.current_file = ""
        self.model = None

        # 检测参数
        self.conf_threshold = 0.7
        self.iou_threshold = 0.5

        # 告警控制
        self.alert_delay_seconds = 20
        self.last_alert_time = 0

        # 邮件配置（从 email_config.json 加载，不存在则初始化为空）
        # 打包后写到 exe 所在目录（可写），开发时写到脚本目录
        if getattr(sys, "frozen", False):
            _cfg_dir = os.path.dirname(sys.executable)
        else:
            _cfg_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(_cfg_dir, "email_config.json")
        self.email_cfg = self.load_email_config()

        # 摄像头状态（枚举在 init_ui 之前完成，供下拉框填充使用）
        self.available_cameras = self.enumerate_cameras()
        self.current_camera_index = None

        # 本地检测模式状态
        self.local_mode = False  # 是否处于本地文件检测模式
        self.local_file = ""  # 当前选中的本地文件路径
        self.local_output_path = ""  # 输出文件路径
        self.video_writer = None  # cv2.VideoWriter（视频模式专用）

        # 初始化UI
        self.init_ui()

        # 摄像头与定时器
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 加载模型
        self.load_model()

        # 启动默认摄像头（取枚举到的第一个，通常是索引 0）
        if self.available_cameras:
            self.switch_camera(self.available_cameras[0])
        else:
            QMessageBox.warning(self, "摄像头", "未检测到任何可用摄像头，请检查设备连接。")

        # 刷新邮件状态标签（init_ui 之后才能调用）
        self._refresh_email_status()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # -------------------------------
        # 左侧：视频显示区域（固定 800x600）
        # -------------------------------
        self.video_label = QLabel("实时检测画面将显示在这里")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("""
            background-color: black;
            border: 2px solid gray;
            color: white;
            font-size: 16px;
        """)

        # -------------------------------
        # 右侧：信息面板
        # -------------------------------
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setSpacing(15)
        info_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("🔥 检测结果")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: red;")
        info_layout.addWidget(title)

        self.lbl_target_count = QLabel("目标数目：0")
        self.lbl_target_count.setFont(QFont("Arial", 14))
        info_layout.addWidget(self.lbl_target_count)

        self.lbl_target_type = QLabel("类型：暂无")
        self.lbl_target_type.setFont(QFont("Arial", 14))
        info_layout.addWidget(self.lbl_target_type)

        self.lbl_confidence = QLabel("置信度：暂无")
        self.lbl_confidence.setFont(QFont("Arial", 14))
        info_layout.addWidget(self.lbl_confidence)

        self.lbl_location = QLabel("位置：暂无")
        self.lbl_location.setFont(QFont("Arial", 14))
        info_layout.addWidget(self.lbl_location)

        # 置信度设置
        conf_label = QLabel("置信度阈值 (0~0.99):")
        self.conf_input = QLineEdit("0.7")
        self.conf_input.setMaxLength(5)
        apply_conf_btn = QPushButton("应用设置")
        apply_conf_btn.clicked.connect(self.on_apply_conf_threshold)

        info_layout.addWidget(conf_label)
        info_layout.addWidget(self.conf_input)
        info_layout.addWidget(apply_conf_btn)

        # -------------------------------
        # 摄像头控制区
        # -------------------------------
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: gray;")
        info_layout.addWidget(separator)

        cam_title = QLabel("摄像头控制")
        cam_title.setFont(QFont("Arial", 13, QFont.Bold))
        info_layout.addWidget(cam_title)

        self.cam_combo = QComboBox()
        self._refresh_combo()
        info_layout.addWidget(self.cam_combo)

        btn_row = QHBoxLayout()
        switch_btn = QPushButton("切换摄像头")
        switch_btn.clicked.connect(self.on_switch_camera)
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.on_refresh_cameras)
        btn_row.addWidget(switch_btn)
        btn_row.addWidget(refresh_btn)
        info_layout.addLayout(btn_row)

        self.lbl_cam_status = QLabel("状态：未启动")
        self.lbl_cam_status.setFont(QFont("Arial", 11))
        self.lbl_cam_status.setStyleSheet("color: gray;")
        info_layout.addWidget(self.lbl_cam_status)

        # -------------------------------
        # 本地检测区
        # -------------------------------
        sep2 = QLabel()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: gray;")
        info_layout.addWidget(sep2)

        local_title = QLabel("本地视频/图片检测")
        local_title.setFont(QFont("Arial", 13, QFont.Bold))
        info_layout.addWidget(local_title)

        self.btn_select_file = QPushButton("检测本地视频/图片")
        self.btn_select_file.clicked.connect(self.on_select_local_file)
        info_layout.addWidget(self.btn_select_file)

        self.lbl_local_file = QLineEdit("未选择文件")
        self.lbl_local_file.setReadOnly(True)
        self.lbl_local_file.setStyleSheet("color: gray;")
        info_layout.addWidget(self.lbl_local_file)

        self.btn_start_detect = QPushButton("开始测试")
        self.btn_start_detect.setEnabled(False)
        self.btn_start_detect.clicked.connect(self.on_start_local_detect)
        info_layout.addWidget(self.btn_start_detect)

        self.lbl_local_status = QLabel("请先选择文件")
        self.lbl_local_status.setFont(QFont("Arial", 11))
        self.lbl_local_status.setStyleSheet("color: gray;")
        self.lbl_local_status.setWordWrap(True)
        info_layout.addWidget(self.lbl_local_status)

        # -------------------------------
        # 邮件告警设置区
        # -------------------------------
        sep3 = QLabel()
        sep3.setFixedHeight(1)
        sep3.setStyleSheet("background-color: gray;")
        info_layout.addWidget(sep3)

        mail_title = QLabel("邮件告警设置")
        mail_title.setFont(QFont("Arial", 13, QFont.Bold))
        info_layout.addWidget(mail_title)

        self.btn_email_settings = QPushButton("邮件设置")
        self.btn_email_settings.clicked.connect(self.on_email_settings)
        info_layout.addWidget(self.btn_email_settings)

        self.lbl_email_status = QLabel("状态：未配置")
        self.lbl_email_status.setFont(QFont("Arial", 11))
        self.lbl_email_status.setStyleSheet("color: gray;")
        self.lbl_email_status.setWordWrap(True)
        info_layout.addWidget(self.lbl_email_status)

        info_layout.addStretch()

        # -------------------------------
        # 主布局
        # -------------------------------
        main_layout.addWidget(self.video_label, 7)
        main_layout.addWidget(info_panel, 3)

        self.setCentralWidget(main_widget)

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ 模型加载成功: {self.model_path}")
            else:
                QMessageBox.warning(self, "错误", f"模型文件不存在: {self.model_path}")
                self.model = None
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {e!s}")
            self.model = None

    def enumerate_cameras(self, max_test=5):
        """扫描索引 0~max_test-1，返回实际可用的摄像头索引列表."""
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def _refresh_combo(self):
        """用 available_cameras 重新填充下拉框."""
        self.cam_combo.clear()
        for idx in self.available_cameras:
            label = f"摄像头 {idx}（{'内置' if idx == 0 else 'USB'}）"
            self.cam_combo.addItem(label, userData=idx)
        self.cam_combo.addItem("关闭摄像头", userData=-1)

    def switch_camera(self, camera_index):
        """切换到指定摄像头；camera_index=-1 表示关闭摄像头."""
        # 若处于本地检测模式，先清理资源
        if self.local_mode:
            self.local_mode = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if camera_index == -1:
            self.current_camera_index = None
            self.video_label.setText("摄像头已关闭")
            self.lbl_cam_status.setText("状态：已关闭")
            self.lbl_cam_status.setStyleSheet("color: gray;")
            return

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            QMessageBox.warning(self, "摄像头错误", f"无法打开摄像头 {camera_index}，请检查设备连接！")
            self.lbl_cam_status.setText(f"状态：摄像头 {camera_index} 打开失败")
            self.lbl_cam_status.setStyleSheet("color: red;")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap = cap
        self.current_camera_index = camera_index
        self.timer.start(30)

        cam_label = f"摄像头 {camera_index}（{'内置' if camera_index == 0 else 'USB'}）"
        self.lbl_cam_status.setText(f"状态：使用{cam_label}")
        self.lbl_cam_status.setStyleSheet("color: green;")
        print(f"✅ 已切换到摄像头 {camera_index}")

    def on_switch_camera(self):
        """「切换摄像头」按钮回调."""
        camera_index = self.cam_combo.currentData()
        if camera_index is None:
            return
        self.switch_camera(camera_index)

    def on_refresh_cameras(self):
        """「刷新列表」按钮回调：重新枚举并更新下拉框."""
        self.available_cameras = self.enumerate_cameras()
        self._refresh_combo()
        count = len(self.available_cameras)
        QMessageBox.information(self, "刷新完成", f"检测到 {count} 个可用摄像头")

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.local_mode:
                self._finalize_video()  # 视频读完，执行收尾
            return

        annotated_frame = frame  # 默认原图
        high_conf_target_exists = False
        results = []  # 防止 model 未加载时下方引用出错

        if self.model is not None:
            try:
                results = self.model(frame, conf=self.conf_threshold, imgsz=640)
                if results and len(results) > 0:
                    r = results[0]
                    annotated_frame = r.plot()  # 带框图像
                    boxes = r.boxes
                    alert_conf = 0.0
                    for box in boxes:
                        conf = float(box.conf.item())
                        if conf >= self.conf_threshold:
                            high_conf_target_exists = True
                            alert_conf = conf
                            break
            except Exception as e:
                print(f"⚠️ 模型推理出错: {e}")
                annotated_frame = frame  # 出错时仍然显示原图

        # 本地视频模式：将当前帧写入输出文件
        if self.local_mode and self.video_writer is not None:
            self.video_writer.write(annotated_frame)

        # ✅ 告警逻辑：仅摄像头模式触发，冷却 20 秒
        if not self.local_mode and high_conf_target_exists:
            current_time = time.time()
            if current_time - self.last_alert_time >= self.alert_delay_seconds:
                sender = self.email_cfg.get("sender", "")
                receiver = self.email_cfg.get("receiver", "")
                password = self.email_cfg.get("password", "")
                if not sender or not receiver or not password:
                    print("⚠️ 邮件未配置，跳过告警。请点击「邮件设置」完成配置。")
                else:
                    print("🔥 检测到高置信目标，发送告警邮件...")
                    import threading

                    conf_to_send = alert_conf
                    frame_to_send = annotated_frame.copy()
                    self.last_alert_time = time.time()  # 立即更新，防止并发重复触发

                    def send_alert():
                        try:
                            mailToMeWithImage(
                                "火灾告警",
                                f"警告：摄像头检测到高置信度可疑目标，当前置信度为 {conf_to_send * 100:.2f}%，可能存在风险，请查看附件！",
                                frame_to_send,
                                sender,
                                receiver,
                                password,
                            )
                            print("✅ 告警邮件发送成功！")
                        except Exception as e:
                            print(f"❌ 邮件发送失败: {e}")

                    threading.Thread(target=send_alert, daemon=True).start()

        # -------------------------------
        # 显示图像（固定为 800x600，防止拉伸）
        # -------------------------------
        try:
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            fixed_width, fixed_height = 800, 600
            resized_frame = cv2.resize(annotated_frame_rgb, (fixed_width, fixed_height))

            h, w, c = resized_frame.shape
            q_img = QImage(resized_frame.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
        except Exception as e:
            print(f"❌ 显示图像出错: {e}")

        # -------------------------------
        # 更新右侧信息（仅第一个目标）
        # -------------------------------
        if self.model is not None and results and len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names.get(cls, f"Class_{cls}")

            self.lbl_target_count.setText("目标数目：1")
            self.lbl_target_type.setText(f"类型：{label}")
            self.lbl_confidence.setText(f"置信度：{conf * 100:.2f}%")
            self.lbl_location.setText(f"位置：xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2}")
        else:
            self.lbl_target_count.setText("目标数目：0")
            self.lbl_target_type.setText("类型：暂无")
            self.lbl_confidence.setText("置信度：暂无")
            self.lbl_location.setText("位置：暂无")

    def on_apply_conf_threshold(self):
        text = self.conf_input.text().strip()
        try:
            new_conf = float(text)
            if 0.0 <= new_conf <= 0.99:
                self.conf_threshold = new_conf
                self.lbl_confidence.setText(f"置信度：暂无（当前阈值：{self.conf_threshold:.2f}）")
                QMessageBox.information(self, "设置成功", f"置信度阈值已更新为：{new_conf:.2f}")
            else:
                QMessageBox.warning(self, "输入错误", "请输入 0.0 ~ 0.99 之间的数值！")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字！")

    # ============================================================
    # 本地检测相关方法
    # ============================================================

    def on_select_local_file(self):
        """弹出文件选择框，让用户选择本地视频或图片."""
        path, _ = QFileDialog.getOpenFileName(
            self, "请选择待检测的本地文件", "", "视频/图片文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self.local_file = path
            self.lbl_local_file.setText(os.path.basename(path))
            self.lbl_local_file.setStyleSheet("color: black;")
            self.btn_start_detect.setEnabled(True)
            self.lbl_local_status.setText("已选择文件，点击开始测试")
            self.lbl_local_status.setStyleSheet("color: blue;")

    def on_start_local_detect(self):
        """「开始测试」按钮回调：按文件类型分发到图片或视频处理."""
        if not self.local_file:
            return
        ext = os.path.splitext(self.local_file)[1].lower()
        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        video_exts = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
        if ext in image_exts:
            self._detect_image(self.local_file)
        elif ext in video_exts:
            self._detect_video(self.local_file)
        else:
            QMessageBox.warning(self, "格式错误", f"不支持的文件格式：{ext}")

    def _detect_image(self, path):
        """图片一次性检测：推理→保存→显示."""
        if self.model is None:
            QMessageBox.warning(self, "错误", "模型未加载！")
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", "图片文件无法读取，请检查文件格式！")
            return
        try:
            results = self.model(frame, conf=self.conf_threshold, imgsz=640)
            annotated = results[0].plot() if results and len(results) > 0 else frame
        except Exception as e:
            QMessageBox.critical(self, "推理错误", f"模型推理失败：{e}")
            return

        # 保存结果
        out_path = self._get_output_path(path, "ppicture_detect", ".jpg")
        cv2.imwrite(out_path, annotated)

        # 在左侧显示结果
        try:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (800, 600))
            h, w, c = resized.shape
            q_img = QImage(resized.data, w, h, w * c, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"❌ 图片显示出错: {e}")

        self.lbl_local_status.setText("检测完成")
        self.lbl_local_status.setStyleSheet("color: green;")
        QMessageBox.information(self, "检测完成", f"结果已保存至：\n{out_path}")

    def _detect_video(self, path):
        """视频检测：复用 cap+timer 机制，逐帧推理并写入输出文件."""
        if self.model is None:
            QMessageBox.warning(self, "错误", "模型未加载！")
            return

        # 停止当前摄像头，清理旧资源
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", "视频文件无法打开，请检查文件格式！")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = self._get_output_path(path, "vvedio_detect", ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        self.local_output_path = out_path

        self.cap = cap
        self.local_mode = True
        self.btn_start_detect.setEnabled(False)

        self.lbl_local_status.setText("正在检测中...")
        self.lbl_local_status.setStyleSheet("color: orange;")
        self.lbl_cam_status.setText("状态：本地视频检测中")
        self.lbl_cam_status.setStyleSheet("color: orange;")
        self.timer.start(30)

    def _finalize_video(self):
        """视频检测结束时的收尾工作."""
        self.timer.stop()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.local_mode = False
        self.btn_start_detect.setEnabled(True)

        out_path = self.local_output_path
        self.lbl_local_status.setText("检测完成")
        self.lbl_local_status.setStyleSheet("color: green;")
        self.lbl_cam_status.setText("状态：检测完成")
        self.lbl_cam_status.setStyleSheet("color: green;")
        QMessageBox.information(self, "检测完成", f"结果已保存至：\n{out_path}")

    def _get_output_path(self, src_path, prefix, ext):
        """生成不冲突的输出文件路径，编号从1开始自动递增."""
        directory = os.path.dirname(os.path.abspath(src_path))
        idx = 1
        while True:
            full = os.path.join(directory, f"{prefix}{idx}{ext}")
            if not os.path.exists(full):
                return full
            idx += 1

    # ============================================================
    # 邮件配置相关方法
    # ============================================================

    def load_email_config(self) -> dict:
        """从 email_config.json 读取邮件配置，文件不存在或解析失败时返回空配置."""
        empty = {"sender": "", "receiver": "", "password": ""}
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"读取邮件配置失败: {e}")
        return empty

    def save_email_config(self, cfg: dict):
        """将邮件配置写入 email_config.json."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存邮件配置失败: {e}")

    def on_email_settings(self):
        """打开邮件设置对话框."""
        dlg = EmailSettingsDialog(self.email_cfg, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.email_cfg = dlg.get_config()
            self.save_email_config(self.email_cfg)
            self._refresh_email_status()
            QMessageBox.information(self, "保存成功", "邮件配置已保存！")

    def _refresh_email_status(self):
        """根据当前 email_cfg 更新右侧状态标签."""
        sender = self.email_cfg.get("sender", "")
        receiver = self.email_cfg.get("receiver", "")
        if sender and receiver:
            self.lbl_email_status.setText(f"已配置：\n{sender}\n→ {receiver}")
            self.lbl_email_status.setStyleSheet("color: green;")
        else:
            self.lbl_email_status.setText("状态：未配置")
            self.lbl_email_status.setStyleSheet("color: gray;")

    def closeEvent(self, event):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv8DetectionApp()
    window.show()
    sys.exit(app.exec_())
