import cv2
import numpy as np
import mediapipe as mp

shape = (768, 432)

ocap = cv2.VideoCapture("./data/out.mp4")
icap = cv2.VideoCapture("./data/vid2.mp4")
out = cv2.VideoWriter(
    "./data/post-estimation.mp4", cv2.VideoWriter_fourcc(*"MPEG"), 30, (864, 768)
)

while True:
    _, qframe = ocap.read()

    _, rframe = icap.read()
    iframe = cv2.resize(rframe, shape[::-1])
    oframe = cv2.resize(qframe, shape[::-1])

    h = np.hstack((iframe, oframe))

    h = cv2.putText(
        h,
        "Original",
        (5, 25),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    h = cv2.putText(
        h,
        "Inferred",
        (432 + 5, 25),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out.write(h)
    cv2.imshow("out", h)
    # cv2.imshow("ocap", oframe)
    # cv2.imshow("icap", iframe)
    if cv2.waitKey(30) & 0xFF == 27:
        break

ocap.release()
icap.release()
cv2.destroyAllWindows()
