import cv2
import numpy as np
import mediapipe as mp
import sys

shape = (768, 432)

if len(sys.argv) != 4:
    print("python download.py [output name] [original] [estimated]")
    sys.exit(1)
ocap = cv2.VideoCapture(sys.argv[2])
icap = cv2.VideoCapture(sys.argv[3])
out = cv2.VideoWriter(
    f"./data/{sys.argv[1]}.mp4", cv2.VideoWriter_fourcc(*"MPEG"), 30, (864, 768)
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
        "Estimated",
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
