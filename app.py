from flask import *
import cv2
import mediapipe
import numpy as np

video = cv2.VideoCapture(0)
drawing = mediapipe.solutions.drawing_utils
solHand = mediapipe.solutions.hands
solPose = mediapipe.solutions.pose
hands = solHand.Hands()
pose = solPose.Pose()
canvas = ""


app = Flask(__name__)

@app.route("/")
def index():
    return "<a href='board'>Start</a>"


@app.route("/board")
def board():
    return Response(play(), mimetype='multipart/x-mixed-replace; boundary=frame')


def find_hands(img, draw):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    lms = []
    if result.multi_hand_landmarks:
        l = []
        for mark in result.multi_hand_landmarks:
            for i, lm in enumerate(mark.landmark):
                l.append(lm)
            if draw:
                drawing.draw_landmarks(img, mark, solHand.HAND_CONNECTIONS)
            lms.append(l)
    return lms


def compute(img, h):
    he, w, c = img.shape
    l = []
    for i in range(1, 6):
        ind = h[getId(i)]
        l.append((int(ind.x*w), int(ind.y*he)))
    return l


def fingers(h):
    l = []
    if h[getId(1)].x < h[getId(1)-1].x:
        l.append(1)
    else:
        l.append(0)
    for i in range(2, 6):
        id = getId(i)
        if h[id].y < h[id-2].y:
            l.append(1)
        else:
            l.append(0)
    return l


def getId(i):
    return [4, 8, 12, 16, 20][i-1]


def play():
    px = py = -1
    global canvas
    while True:
        _, img = video.read()
        img = cv2.flip(img, 1)
        if len(canvas) == 0:
            canvas = np.zeros(img.shape, np.uint8)
        lms = find_hands(img, draw=True)
        if len(lms) > 0:
            h1 = lms[0]
            c = compute(img, h1)
            f = fingers(h1)
            if f[1] and f[2] == False:
                if px == -1 and py == -1:
                    px, py = c[1]
                cv2.circle(img, c[1], 10, (255, 50, 255), 10, cv2.FILLED)
                cv2.line(canvas, (px, py), c[1],
                         (0, 255, 100), 10, cv2.FILLED)
                px, py = c[1]

            if f[1] and f[2]:
                px = py = -1

            if sum(f) == 5 or sum(f[1:]) == 4:
                cv2.circle(img, c[1], 70, (255, 255, 255), cv2.FILLED)
                cv2.circle(canvas, c[1], 70, (0, 0, 0),  cv2.FILLED)

        img = cv2.addWeighted(img, 0.9, canvas, 0.9, 0.7)
        ret, buffer = cv2.imencode(".jpg", img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')


if __name__ == "__main__":
    app.run()
