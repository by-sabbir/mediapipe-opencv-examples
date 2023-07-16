import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from helpers import draw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BASE_PATH = Path.cwd()
MODEL_PATH = BASE_PATH / "tasks"
THUMB_TIP = 4
INDEX_FINGER_TIP = 8

pose_model_path = MODEL_PATH / "hand-landmark.task"

base_options = python.BaseOptions(model_asset_path=pose_model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

src = cv2.imread("./data/hand-pinch.png", cv2.IMREAD_COLOR)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=src)

result = detector.detect(mp_image)


hand_landmarks_list = result.hand_landmarks[0]
handness_list = result.handedness


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


height, width, _ = src.shape
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

rendered_img = cv2.line(src, (thumb_x, thumb_y), (index_x, index_y), (255,255,255), 2)
rendered_img = cv2.putText(rendered_img, f"{int(d)}", (index_x, index_y - 20), 1, 1.2, (0, 0, 255), 1)

cv2.imshow("out", rendered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
