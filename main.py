import os
import time
import numpy as np
import json
import cv2
import math
from ultralytics import YOLO
import torch
from PIL import Image

print(f"torch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# cuda_driver_ver = cudaDriverGetVersion()
# print(f"CUDA Driver Version: {cuda_driver_ver}")

def save_frame(frame, frame_number):
    color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to PIL Image format
    pil_image = Image.fromarray(color_coverted)

    # Save the frame as an image file
    frame_path = f"{output_path}/frame_{str(frame_number).zfill(12)}.jpg"
    pil_image.save(frame_path)


def calculate_real_ball(results):
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Class probabilities for classification outputs

        # check class_id, sports ball is 32
        class_id = np.array(boxes.cls.cpu(), dtype="int")
        # print(class_id)

        indices = [i for i, item in enumerate(class_id) if item == 32]
        if len(indices) == 0:
            return None
        if len(indices) > 1:
            cv2.waitKey(0)
            ## disregard with area of ball bbox, alternatively, length of mask xy list
            ball_area_list = []
            for i, ii in enumerate(indices):
                ball_xywh = (np.array(boxes[ii].xywh.tolist())[0])
                ball_area = ball_xywh[2] * ball_xywh[3]
                ball_area_list.append(ball_area)
            iind = ball_area_list.index(max(ball_area_list))
            ind = indices[iind]
        else:
            ind = indices[0]
        print(f'ball index: {ind}')
    return ind


vid = "./sample_videos/3min.mp4"
output_path = f"output_frames/{vid.split('/')[-1].split('.')[0]}"

# create output folder
if not os.path.exists(output_path):
    print("output frames to", output_path)
    os.makedirs(output_path)


# Load a model
model = YOLO('yolov8l-seg.pt')  # pretrained YOLOv8n model

cap = cv2.VideoCapture(vid)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
frame_number = 0
start_time = time.time()
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("read frame fail")
        break


    results = model.predict(frame, device=device, classes=[0,32], conf=0.3, show_conf=True, iou=0.5)
    real_ball_ind = calculate_real_ball(results)
    frame = results[0].plot(font_size=1, line_width=1)


    cv2.putText(frame, str(frame_number), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)  # frame number
    cv2.imshow('frame', frame)

    frame_number += 1
    save_frame(frame, frame_number)

    # mid video ckpt
    if frame_number == 500:
        print(time.time()-start_time)
        cv2.waitKey(0)

    # if len(results[0].boxes) == 0:
    #     cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
