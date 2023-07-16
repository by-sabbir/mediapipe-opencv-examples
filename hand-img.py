import cv2
import mediapipe as mp

from pathlib import Path
from helpers import draw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_PATH = Path.cwd()
MODEL_PATH = BASE_PATH / "tasks"

pose_model_path = MODEL_PATH / "hand-landmark.task"

base_options = python.BaseOptions(model_asset_path=pose_model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

src = cv2.imread("./data/hand1.jpg", cv2.IMREAD_COLOR)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=src)

result = detector.detect(mp_image)

rendered_img = draw.draw_hand_landmarks_on_image(src, result)
cv2.imshow("out", rendered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
