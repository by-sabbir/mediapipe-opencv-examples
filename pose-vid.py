import cv2
import sys
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from helpers import draw

BASE_PATH = Path.cwd()
MODEL_PATH = BASE_PATH / "tasks"

pose_model_path = MODEL_PATH / "pose-landmark.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO,
)

cv2.namedWindow("pose-vid", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("./data/vid2.mp4")
if not cap.isOpened():
    print("could not play from provided file")
    sys.exit(124)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    "./data/out.mp4", cv2.VideoWriter_fourcc(*"MPEG"), fps, (432, 769)
)
print("fps(s): ", fps)
delta = int((1.0 / fps) * 1000)
timestamp = int(datetime.now().timestamp())
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame_orig = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame_orig, (432, 769))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp += delta
        pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
        # print(pose_landmarker_result)
        rendered_img = draw.draw_landmarks_on_image(frame, pose_landmarker_result)
        out.write(rendered_img)
        cv2.imshow("pose-vid", rendered_img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
