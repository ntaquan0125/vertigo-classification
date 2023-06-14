import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from model import *
from UI import UI
from utils import *


scaler = StandardScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D(5, 6).to(device)
model.load_state_dict(torch.load('models/model_0.pth', map_location=torch.device('cpu')))


class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)
    position_signal = QtCore.pyqtSignal(np.ndarray)
    classification_signal = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture('./dataset/Rt_Geo_BPPV/Class3_100063.avi')
        mat = loadmat('./dataset/Rt_Geo_BPPV/Class3_100063.mat')

        right_data = iter(np.load('bppv/right1.npy'))
        left_data = iter(np.load('bppv/left1.npy'))
        right_bbox = [0] * 3
        left_bbox = [0] * 3
        process_buf = []
        ids_buf = []

        while True:
            ret, frame = cap.read()

            if ret:
                right, left = crop_frame(frame)
                temp_bbox = next(right_data)
                if not detect_blink(right, right_bbox):
                    right_bbox = temp_bbox
                    right = draw_bbox(right, right_bbox)
                temp_bbox = next(left_data)
                if not detect_blink(left, left_bbox):
                    left_bbox = temp_bbox
                    left = draw_bbox(left, left_bbox)
                process_buf.append(np.stack((right_bbox, left_bbox)))

                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                for i in range(0, len(mat['fr']), 2):
                    if frame_id > mat['fr'][i] and frame_id < mat['fr'][i + 1]:
                        ids_buf.append(int(mat['label'][i]))
                        break
                    if i == len(mat['fr']) - 2:
                        ids_buf.append(-1)

                if len(process_buf) == 100:
                    data = np.array(process_buf).swapaxes(0, 1)
                    data = preprocess_timeseries_data(data)
                    data[0] = scaler.fit_transform(data[0])
                    data[1] = scaler.fit_transform(data[1])

                    try:
                        data = torch.FloatTensor(data).transpose(1, 2).to(device)
                        lr = torch.Tensor([[0], [1]]).to(device)
                        action = max(set(filter(lambda x: x != -1, ids_buf)), key = ids_buf.count)
                        action = F.one_hot(torch.Tensor([action - 1]).long(), 7).expand(2, -1).to(device)

                        outputs = model(data, lr, action)
                        outputs = F.softmax(outputs, dim=1)
                        self.classification_signal.emit(outputs.cpu().detach().numpy())
                    except ValueError:
                        pass

                    del process_buf[: 50]
                    del ids_buf[: 50]

                self.change_pixmap_signal.emit(np.stack((right, left)))
                self.position_signal.emit(np.stack((right_bbox, left_bbox)))
            time.sleep(1/24)


class App(UI):
    def __init__(self):
        super(App, self).__init__()
        self.functions_init()

    def functions_init(self):
        # Plotting
        BUF_LEN = 200
        self.t = -BUF_LEN
        self.xr_buf = np.zeros((BUF_LEN))
        self.xl_buf = np.zeros((BUF_LEN))
        self.yr_buf = np.zeros((BUF_LEN))
        self.yl_buf = np.zeros((BUF_LEN))

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.position_signal.connect(self.update_plot)
        self.thread.classification_signal.connect(self.update_progress_bar)
        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_right_img = self.convert_cv_qt(cv_img[0])
        qt_left_img = self.convert_cv_qt(cv_img[1])
        self.right_img.setPixmap(qt_right_img)
        self.left_img.setPixmap(qt_left_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    @QtCore.pyqtSlot(np.ndarray)
    def update_plot(self, position):
        try:
            self.xr_buf[:-1] = self.xr_buf[1:]
            self.xl_buf[:-1] = self.xl_buf[1:]
            self.yr_buf[:-1] = self.yr_buf[1:]
            self.yl_buf[:-1] = self.yl_buf[1:]
            self.xr_buf[-1] = position[0][0]
            self.xl_buf[-1] = position[1][0]
            self.yr_buf[-1] = position[0][1]
            self.yl_buf[-1] = position[1][1]
            self.t += 1
        except IndexError:
            pass
        finally:
            self.xr_curve.setData(self.xr_buf)
            self.xr_curve.setPos(self.t, 0)
            self.xl_curve.setData(self.xl_buf)
            self.xl_curve.setPos(self.t, 0)
            self.yr_curve.setData(self.yr_buf)
            self.yr_curve.setPos(self.t, 0)
            self.yl_curve.setData(self.yl_buf)
            self.yl_curve.setPos(self.t, 0)
            QtWidgets.QApplication.processEvents()

    @QtCore.pyqtSlot(np.ndarray)
    def update_progress_bar(self, logits):
        classes = ["Lt_Apo_BPPV", "Lt_Geo_BPPV", "Lt_PC_BPPV", "Rt_Apo_BPPV", "Rt_Geo_BPPV", "Rt_PC_BPPV"]
        sum_logits = np.sum(logits, 0)
        pred = logits[:, np.argmax(sum_logits, 0)]
        self.label.setText(f'Diagnosis: \n{classes[np.argmax(sum_logits, 0)]}')
        self.confident_bar.setValue(int((pred[0] + pred[1]) * 50))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())