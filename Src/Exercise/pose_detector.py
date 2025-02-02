import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation)
        self.mp_draw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        lmList = self.findPosition(img, draw=False)
        if lmList:
            x1, y1 = lmList[p1][1], lmList[p1][2]
            x2, y2 = lmList[p2][1], lmList[p2][2]
            x3, y3 = lmList[p3][1], lmList[p3][2]

            # Calculate the angle
            angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
            angle = angle + 360 if angle < 0 else angle
            angle = angle if angle <= 180 else 360 - angle

            if draw:
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return angle
        return 0
