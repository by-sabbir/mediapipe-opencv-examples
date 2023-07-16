import cv2
import mediapipe as mp

from pathlib import Path

from helpers import draw


cv2.namedWindow("pose", cv2.WINDOW_NORMAL)
BASE_PATH = Path.cwd()
MODEL_PATH = BASE_PATH / "tasks"

pose_model_path = MODEL_PATH / "pose-landmark.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.IMAGE,
)


orig = cv2.imread("data/yoga1.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(orig, (480, 480))

mp_img = mp.Image(mp.ImageFormat.SRGB, data=img)

result = None
with PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_img)
    rendered_img = draw.draw_landmarks_on_image(img, result)
    cv2.imshow("pose", rendered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
