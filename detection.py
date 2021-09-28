import cv2
import mediapipe
import numpy as np

video = cv2.VideoCapture(0)
face = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
drawing = mediapipe.solutions.drawing_utils
solHand = mediapipe.solutions.hands
solPose = mediapipe.solutions.pose
hands = solHand.Hands()
pose = solPose.Pose()
canvas = ""


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


def find_pose(img, draw):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    lms = []
    if result.pose_landmarks:
        if draw:
            drawing.draw_landmarks(
                img, result.pose_landmarks, solPose.POSE_CONNECTIONS)
    return lms


def distance(a, b):
    return (((a[0]-b[0])**2)+((a[1]-b[1])**2))**(1/2)


def inDistance(img, h1):
    ind = h1[8]
    tum = h1[4]
    if ind.y < h1[6].y:
        h, w, c = img.shape
        v = (int(ind.x*w), int(ind.y*h))
        u = (int(tum.x*w), int(tum.y*h))
        cv2.circle(img, v, 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, u, 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, v, u, (0, 255, 255))
        cv2.putText(img, str(int(distance(v, u))), (4, 130),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255))
    else:
        pass


def getId(i):
    return [4, 8, 12, 16, 20][i-1]


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


def printNum(img, h):
    f = fingers(h)
    if f[2] == 1 and sum(f) == 1:
        cv2.putText(img, "F**K OFF", (4, 130),
                    cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 255))
    else:
        cv2.putText(img, str(sum(f)), (4, 130),
                    cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 255))


def compute(img, h):
    he, w, c = img.shape
    l = []
    for i in range(1, 6):
        ind = h[getId(i)]
        l.append((int(ind.x*w), int(ind.y*he)))
    return l


if __name__ == "__main__":
    px = py = -1
    while True:
        _, img = video.read()
        img = cv2.flip(img, 1)
        if len(canvas) == 0:
            canvas = np.zeros(img.shape, np.uint8)
        lms = find_hands(img, draw=True)
        # plms = find_pose(img, draw=True)
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
                cv2.circle(img, c[1], 100, (255, 255, 255), cv2.FILLED)
                cv2.line(canvas, (px, py), c[1], (0, 0, 0), 60, cv2.FILLED)

        img = cv2.addWeighted(img, 0.9, canvas, 0.9, 0.7)
        cv2.imshow("photo", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
