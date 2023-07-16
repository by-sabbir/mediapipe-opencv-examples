import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from helpers import draw

THUMB_TIP = 4
INDEX_FINGER_TIP = 8
BASE_PATH = Path.cwd()
MODEL_PATH = BASE_PATH / "tasks"

pose_model_path = MODEL_PATH / "hand-landmark.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO,
)

def calculate_distance(detection_result, img_shape):
    hand_landmarks_list = detection_result.hand_landmarks[0]
    handness_list = detection_result.handedness

    print(handness_list, type(handness_list))

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks_list
                ]
            )

    landmarks = hand_landmarks_proto.landmark
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]


    height, width, _ = img_shape
    print("thumb\n")
    print(thumb_tip)

    print("index\n")
    print(index_tip)

    thumb_x = int(thumb_tip.x * width)
    thumb_y = int(thumb_tip.y * height)

    index_x = int(index_tip.x * width)
    index_y = int(index_tip.y * height)

    npx = np.array([thumb_x, thumb_y])
    npy = np.array([index_x, index_y])

    d = np.linalg.norm(npx - npy)

    return d, (thumb_x, thumb_y), (index_x, index_y)


def set_volume(dist, max_px=164):
    vol = int((dist/max_px) * 100)
    cmd = f"amixer sset 'Master' {vol}%"
    out = os.popen(cmd).read()

    print(out)


cv2.namedWindow("handpose-vid", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("./data/hand-gesture.m4v")
if not cap.isOpened():
    print("could not play from provided file")
    sys.exit(124)
fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter(
#     "./data/out.mp4", cv2.VideoWriter_fourcc(*"MPEG"), fps, (432, 769)
# )
print("fps(s): ", fps)
delta = int((1.0 / fps) * 1000)
timestamp = int(datetime.now().timestamp())

distx = []
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame_orig = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame_orig, (768, 432))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp += delta
        pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
        
        try:
            dist, thmb, idx = calculate_distance(pose_landmarker_result, img_shape=frame.shape)
            print("distance: ", dist)
            distx.append(dist)

            rendered_img = cv2.line(frame, thmb, idx, (255,255,255), 2)
            rendered_img = cv2.putText(rendered_img, str(dist), (10, 50), 1, 1.2, (255, 255, 255), 1)
            set_volume(dist)
        except:
            continue
        # out.write(rendered_img)
        cv2.imshow("handpose-vid", rendered_img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
