import cv2
import numpy as np


def crop_frame(frame):
    if len(frame.shape) == 2:
        w, h = frame.shape
    else:
        w, h, c = frame.shape

    right = frame[0: w // 2, 0: h // 2]
    left = frame[0: w // 2, h // 2:]
    return right, left


def detect_blink(img, bbox):
    bbox = [int(i) for i in bbox]
    roi = img[bbox[1] - bbox[2] // 3: bbox[1] + bbox[2] // 3, bbox[0] - bbox[2] // 3: bbox[0] + bbox[2] // 3, :]
    return np.mean(roi) > 150


def preprocess_timeseries_data(data):
    vx = np.diff(data[:, :, 0], append=0)[..., np.newaxis]
    vy = np.diff(data[:, :, 1], append=0)[..., np.newaxis]
    data = np.concatenate((data, vx, vy), axis=-1)
    return data


def draw_bbox(img, bbox):
    temp = img.copy()
    cv2.drawMarker(temp, (int(bbox[0]), int(bbox[1])), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    temp = cv2.circle(temp, (int(bbox[0]), int(bbox[1])), int(bbox[2] / 2), (0, 255, 0), 1)
    return temp