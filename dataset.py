import os

import cv2
import numpy as np

from glob import glob
from scipy.io import loadmat


def crop_frame(frame):
    if len(frame.shape) == 2:
        w, h = frame.shape
    else:
        w, h, c = frame.shape

    left = frame[0:w // 2, 0:h // 2]
    right = frame[0:w // 2, h // 2:]
    return left, right


paths = [
    'Lt_Apo_BPPV/Class4_100022',
    'Lt_Geo_BPPV/Class2_100030',
    'Lt_PC_BPPV/Class6_100056',
    'Rt_Apo_BPPV/Class5_100039',
    'Rt_Geo_BPPV/Class3_100063',
    'Rt_PC_BPPV/Class7_100040'
]

for path in paths:
    cap = cv2.VideoCapture('./dataset/' + path + '.avi')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mat = loadmat('./dataset/' + path + '.mat')
    data = list(zip(mat['label'], mat['fr']))

    print(mat)

    n = 0
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(f'./processed_data/' + path + f'r_{data[0][0][0]}_{n:02d}.avi', fourcc, 30, (width // 2, height // 2))

    for i in range(0, len(data) - 1):
        frame_id = 0
        start_id = data[i][1][0]
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_id + frame_id)
            ret, frame = cap.read()
            if not ret:
                break

            frame, _ = crop_frame(frame)
            if i % 2 == 0:
                out.write(frame)


            frame_id += 1

            if frame_id % 100 == 0:
                out.release()
                n += 1
                if i % 2 == 0:
                    out = cv2.VideoWriter(f'./processed_data/' + path + f'r_{data[i][0][0]}_{n:02d}.avi', fourcc, 30, (width // 2, height // 2))
                # else:
                #     out = cv2.VideoWriter(f'./processed_data/Rt_Apo_BPPV/Class5_100039_{0}_{n:02d}.avi', fourcc, 10, (width // 2, height // 2))

            if start_id + frame_id == data[i + 1][1][0]:
                if frame_id >= 100:
                    break
    out.release()


    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(f'./processed_data/' + path + f'l_{data[0][0][0]}_{n:02d}.avi', fourcc, 30, (width // 2, height // 2))

    for i in range(0, len(data) - 1):
        frame_id = 0
        start_id = data[i][1][0]
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_id + frame_id)
            ret, frame = cap.read()
            if not ret:
                break

            _, frame = crop_frame(frame)
            if i % 2 == 0:
                out.write(frame)

            frame_id += 1

            if frame_id % 100 == 0:
                out.release()
                n += 1
                if i % 2 == 0:
                    out = cv2.VideoWriter(f'./processed_data/' + path + f'l_{data[i][0][0]}_{n:02d}.avi', fourcc, 30, (width // 2, height // 2))
                # else:
                #     out = cv2.VideoWriter(f'./processed_data/Rt_Apo_BPPV/Class5_100039_{0}_{n:02d}.avi', fourcc, 10, (width // 2, height // 2))

            if start_id + frame_id == data[i + 1][1][0]:
                if frame_id >= 100:
                    break
    out.release()
cap.release()
cv2.destroyAllWindows()