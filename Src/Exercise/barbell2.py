import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

cap = cv2.VideoCapture(0)


def ask_reps():
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askinteger("Input", "How many reps would you like to do?")
    root.destroy()
    return user_input


# Get desired reps from user
desired_reps = ask_reps()
if desired_reps is None:
    print("No input provided. Exiting...")
    exit()



# Curl counter variables
counter = 0
stage = None
form_feedback = "Stand straight"
display_stand_straight = True

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for left side
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates for right side
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Calculate angle percentage
            left_angle_percentage = np.interp(left_angle, (30, 160), (100, 0))
            right_angle_percentage = np.interp(right_angle, (30, 160), (100, 0))

            # Visualize angles and percentages
            left_angle_color = (0, 255, 0) if left_angle > 160 or left_angle < 30 else (0, 0, 255)
            right_angle_color = (0, 255, 0) if right_angle > 160 or right_angle < 30 else (0, 0, 255)

            cv2.putText(image, f'Left Angle: {int(left_angle)}',
                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_angle_color, 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Angle: {int(right_angle)}',
                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_angle_color, 2, cv2.LINE_AA)

            cv2.putText(image, f'Left: {int(left_angle_percentage)}%',
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, left_angle_color, 2, cv2.LINE_AA)
            cv2.putText(image, f'Right: {int(right_angle_percentage)}%',
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, right_angle_color, 2, cv2.LINE_AA)

            # Curl counter logic for both hands together
            if left_angle > 160 and right_angle > 160:
                stage = "down"
            if left_angle < 30 and right_angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(f'Counter: {counter}')
                form_feedback = "Good form"
            elif abs(left_angle - right_angle) > 20:  # Adjust the threshold as needed
                form_feedback = "Correct your form"
            else:
                form_feedback = "Good form"

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Count: {counter}',
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Form feedback
        if form_feedback:
            cv2.putText(image, form_feedback,
                        (320, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if form_feedback == "Correct your form" else (0, 255, 0), 2, cv2.LINE_AA)

        # Stand straight message
        if display_stand_straight and counter == 0:
            cv2.putText(image, "Stand straight",
                        (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            display_stand_straight = False

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)


        if counter >= desired_reps:
            messagebox.showinfo("Info", "Exercise complete! Great job!")
            break


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()