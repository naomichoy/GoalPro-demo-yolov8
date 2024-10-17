import os
import time
from collections import defaultdict

import numpy as np
import json
import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

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
    frame_path = f"{frame_output_path}/frame_{str(frame_number).zfill(12)}.jpg"
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


# define model
model_size = 'l'
model_task = ''   # -seg, -pose

vid = "./sample_videos/3min.mp4"
frame_output_path = f"output_frames/{vid.split('/')[-1].split('.')[0]}"
vid_output_path = f'output_video'

# create output folder
if not os.path.exists(frame_output_path):
    print("output frames to", frame_output_path)
    os.makedirs(frame_output_path)

if not os.path.exists(vid_output_path):
    print("output vid to", vid_output_path)
    os.makedirs(vid_output_path)

# Load a model
model = YOLO(f'model/yolov8{model_size}{model_task}.pt')  # pretrained YOLOv8n model

# read video input
cap = cv2.VideoCapture(vid)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
frame_number = 0

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video with the specified properties
out = cv2.VideoWriter(f"demo-yolov8{model_size}{model_task}.mp4", cv2.VideoWriter_fourcc(*"MPV4"), fps, (w, h))

# Store the track history
track_history = defaultdict(lambda: [])
draw_track = False

start_time = time.time()
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        print("read frame fail")
        break

    results = model.track(frame, device=device, classes=[0,32], conf=0.28, show_conf=True, iou=0.6, persist=True)

    # filter noisy ball detection
    real_ball_ind = calculate_real_ball(results)

    # draw on frame
    frame = results[0].plot(font_size=1, line_width=1)

    # ## Create an annotator object to draw on the frame
    # annotator = Annotator(frame, line_width=2)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Check if tracking IDs and masks are present in the results
    # if results[0].boxes.id is not None and results[0].masks is not None:
    #     # Extract masks and tracking IDs
    #     masks = results[0].masks.xy
    #     track_ids = results[0].boxes.id.int().cpu().tolist()
    #
    #     # Annotate each mask with its corresponding tracking ID and color
    #     for mask, track_id in zip(masks, track_ids):
    #         annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))

    if draw_track:
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)


    # add frame number to frame
    cv2.putText(frame, str(frame_number), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 1, cv2.LINE_AA)

    # display frame
    cv2.imshow('frame', frame)

    frame_number += 1
    # save frame as picture
    save_frame(frame, frame_number)
    # Write the annotated frame to the output video
    out.write(frame)

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
