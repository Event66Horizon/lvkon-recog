import sys
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from pylibdmtx.pylibdmtx import decode
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QDialog
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
import threading

#     binaries=[('E:\\ANACONDA\\envs\\test\\Lib\\site-packages\\pylibdmtx\\libdmtx-64.dll', '.')],
class MissDialog(QDialog):
    """弹出框：显示标记了 miss 的图像"""
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("不合格")
        layout = QVBoxLayout()
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        self.setLayout(layout)
        self.resize(pixmap.width(), pixmap.height())

class CameraApp(QWidget):
    show_miss_dialog_signal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.initUI()

        self.cap = None
        # self.model = YOLO(model=r'.\\best.pt')
        model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.model = YOLO(model=model_path)

        self.start_time = time.time()
        self.last_qr_data = None
        self.last_frame = None
        self.change_start_time = None
        self.dm_detection_start_time = None
        self.dm_detected = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.skip_detection = False

        self.show_miss_dialog_signal.connect(self.show_miss_dialog)

    def initUI(self):
        self.setWindowTitle('YOLO Inference')
        self.setGeometry(100, 100, 1920, 1080)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(1920, 1080)
        self.video_label.setStyleSheet("border: 2px solid black; margin: 0px; padding: 0px;")

        self.status_label = QLabel('等待摄像头启动...', self)
        self.status_label.setFont(QFont('Arial', 24))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 15px;")

        self.green_light = QLabel(self)
        self.red_light = QLabel(self)
        self.green_light.setFixedSize(50, 50)
        self.red_light.setFixedSize(50, 50)
        self.green_light.setStyleSheet("background-color: green; border-radius: 25px;")
        self.red_light.setStyleSheet("background-color: grey; border-radius: 25px;")

        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label, alignment=Qt.AlignCenter)
        vbox.addWidget(self.status_label, alignment=Qt.AlignCenter)

        hbox = QHBoxLayout()
        hbox.addWidget(self.green_light)
        hbox.addWidget(self.red_light)
        hbox.setAlignment(Qt.AlignCenter)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.show_placeholder_image()

    def showEvent(self, event):
        """窗口实际显示后再启动摄像头等资源。"""
        super().showEvent(event)
        self.startCamera()

    def startCamera(self):
        """在此处初始化摄像头并启动定时器,确保UI已显示后才开始识别流程。"""
        if self.cap is None:
            # self.cap = cv2.VideoCapture(1)
            self.cap = cv2.VideoCapture(0)
            print("摄像头已启动")
            # 打印摄像头支持的分辨率
            for i in range(20):
                width = self.cap.get(i)
                height = self.cap.get(i)
                print(f"{i}: {width}x{height}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3840)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
            print(f"当前分辨率: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        if not self.cap.isOpened():
            self.status_label.setText("无法打开摄像头")
        else:
            self.timer.start(40)  # 每33ms更新一次帧
            self.status_label.setText("摄像头已启动,等待画面稳定...")

    def show_placeholder_image(self):
        # placeholder_image_path = r".\\OIP-C.jpg"
        placeholder_image_path = os.path.join(os.path.dirname(__file__), "OIP-C.jpg")
        placeholder_image = cv2.imread(placeholder_image_path)
        print(placeholder_image_path)
        print("占位图片加载成功")
        if placeholder_image is not None:
            rgb_image = cv2.cvtColor(placeholder_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(1920, 1080, Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(p))
            print("占位图片显示成功")
        else:
            self.status_label.setText("无法加载占位图片")

    def update_frame(self):
        if self.cap is None:
            print("摄像头未启动")
            return
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("无法读取帧")
            return
        
        # 旋转帧
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if self.last_frame is not None:
            edges_current = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 111, 222)
            edges_last = cv2.Canny(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY), 111, 222)
            diff = cv2.absdiff(edges_current, edges_last)
            non_zero_count = np.count_nonzero(diff)
        else:
            non_zero_count = 0  # 初始化时视为稳定
            print("初始化时视为稳定, non_zero_count = 0")

        stable_threshold = 480000
        is_stable = non_zero_count < stable_threshold

        if is_stable and not self.dm_detected and not self.skip_detection:
            if self.dm_detection_start_time is None:
                self.dm_detection_start_time = time.time()
                self.status_label.setText("已稳定,开始识别DM码...")

            elapsed = time.time() - self.dm_detection_start_time
            # 当画面稳定且连续0.75秒时开始识别DM码
            if elapsed >= 0.75:
                threading.Thread(target=self.capture_and_process_frame, args=(frame,)).start()
                self.dm_detected = True
        elif not is_stable:
            if non_zero_count > 800000:
                self.status_label.setText("检测到正在更换电路板")
                self.skip_detection = True
                self.green_light.setStyleSheet("background-color: grey; border-radius: 25px;")
                self.red_light.setStyleSheet("background-color: red; border-radius: 25px;")
            if non_zero_count > 540000:
                self.dm_detection_start_time = None
                self.dm_detected = False
                self.skip_detection = False
                self.status_label.setText("画面不稳定,请保持稳定")

        self.last_frame = frame.copy()

        height, width = frame.shape[:2]
        x = (5 * width) // 12
        w_ = (width) // 6
        y = (height) // 18
        h_ = (2 * height) // 11
        # 在视频帧上绘制矩形，指示二维码应放置的位置
        cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 0, 255), 10)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1920, 1080, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))
    
    # 阈值函数
    def iterative_threshold(self, img_region, initial_T=128):
        """动态迭代法计算最佳阈值"""
        T = initial_T
        while True:
            below = img_region[img_region < T]
            above = img_region[img_region >= T]
            # 如果某一部分像素为空，则无法继续迭代
            if below.size == 0 or above.size == 0:
                break
            v1 = below.mean()
            v2 = above.mean()
            T_new = int((v1 + v2) // 2)
            if T_new == T:
                break
            T = T_new
        return T

    def capture_and_process_frame(self, frame):
        # 保存图像
        # timestamp = str(int(time.time()))
        # img_path = os.path.join(CAPTURE_FOLDER, f"{timestamp}.jpg")
        # cv2.imwrite(img_path, frame)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1920, 1080, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

        # 从图像中提取ROI
        height, width = frame.shape[:2]
        x = (5 * width) // 12
        w_ = (width) // 6
        y = (height) // 18
        h_ = (2 * height) // 11
        roi = frame[y:y + h_, x:x + w_]

        # 转换到 HSV 颜色空间
        # hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # # 定义白色的 HSV 范围
        # lower_white = np.array([0, 0, 200])
        # upper_white = np.array([180, 30, 255])

        # # 创建掩码
        # mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # # 应用掩码提取白色区域
        # white_areas = cv2.bitwise_and(roi, roi, mask=mask)

        # cv2.imshow('White Areas', white_areas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 为了减小背景影响，先对原图像按水平、垂直方向各三等分，取中间靠左侧的区域计算阈值
        h_part = gray.shape[0] // 3
        w_part = gray.shape[1] // 3
        region_for_threshold = gray[0:h_part, w_part:2*w_part]

        best_T = self.iterative_threshold(region_for_threshold)

        # 使用求得的阈值对整个 ROI 进行二值化
        _, binary_roi = cv2.threshold(gray, best_T, 255, cv2.THRESH_BINARY)

        time_start = time.time()

        # 识别 DM 码,超时时间为6秒
        dm_code = decode(binary_roi,timeout=6000)

        time_end = time.time()
        print("Time taken: ", time_end - time_start)

        # dm_code = send_path_to_c(img_path)
        if dm_code:
            identity = dm_code[0].data.decode('utf-8')
            print(f"识别到DM码: {identity}")
            self.status_label.setText(f"识别到DM码: {identity}")
            # 信号灯变绿
            self.green_light.setStyleSheet("background-color: green; border-radius: 25px;")
            self.red_light.setStyleSheet("background-color: grey; border-radius: 25px;")

            # 如果和上次识别到的板子的身份不同，则开始检测miss
            if identity != self.last_qr_data:
                self.last_qr_data = identity
                self.skip_detection = False
                self.status_label.setText(f"电路板已更换,DM码:{identity},开始检测miss...")
                # 传递给miss检测线程
                threading.Thread(target=self.perform_detection, args=(frame,)).start()
            else:
                # 如果和上次识别到的板子的身份相同，则提示更换电路板
                # 同时不进行miss检测
                self.skip_detection = True
                self.status_label.setText("DM码和上次相同,请更换电路板")
        else:
            self.status_label.setText("未检测到DM码")
            self.green_light.setStyleSheet("background-color: grey; border-radius: 25px;")
            self.red_light.setStyleSheet("background-color: red; border-radius: 25px;")

    def perform_detection(self, frame):
        start_time = time.time()
        miss_counts = []
        confidences = []

        while time.time() - start_time < 3:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("无法读取帧")
                return

            results = self.model.predict(source=frame, imgsz=640, conf=0.15, iou=0.45, device='')
            miss_count = 0
            confidences = []
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                if 'miss' in results[0].names[class_id]:
                    confidences.append(box.conf)
                    if box.conf > 0.18:
                        miss_count += 1

            miss_counts.append(miss_count)

        # avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_miss_count = sum(miss_counts) / len(miss_counts) if miss_counts else 0

        annotated_frame = results[0].plot()

        annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_90_CLOCKWISE)

        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1920, 1080, Qt.KeepAspectRatio)
        pixmap_annotated = QPixmap.fromImage(p)
        self.video_label.setPixmap(QPixmap.fromImage(p))

        if avg_miss_count > 0.3:
            self.status_label.setText("检测到miss,不合格")
            self.green_light.setStyleSheet("background-color: grey; border-radius: 25px;")
            self.red_light.setStyleSheet("background-color: red; border-radius: 25px;")
            self.show_miss_dialog_signal.emit(pixmap_annotated)
            time.sleep(1.5)
        else:
            self.status_label.setText("合格")
            self.green_light.setStyleSheet("background-color: green; border-radius: 25px;")
            self.red_light.setStyleSheet("background-color: grey; border-radius: 25px;")
            time.sleep(1.5)

        self.dm_detection_start_time = None
        self.dm_detected = False

    def show_miss_dialog(self, pixmap_annotated):
        dialog = MissDialog(pixmap_annotated, self)
        dialog.show()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CameraApp()
    ex.show()
    sys.exit(app.exec_())