import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to get the target number of reps using tkinter
def get_target_reps():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    reps = simpledialog.askinteger("Input", "How many reps do you want to do?",
                                   minvalue=1, maxvalue=100)
    root.destroy()
    return reps


# Initialize the tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

# Get the target number of reps
target_reps = get_target_reps()
if target_reps is None:
    print("No target reps provided. Exiting...")
    exit()

# Initialize variables for counting reps and tracking progress
counter = 0
stage = {"left": None, "right": None}
form_feedback = "Good Form"

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set initial window size
window_name = 'Pose Detection with Reps and Form Feedback'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 900)  # Set the window size (width, height)

# Setup MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks for both sides
        try:
            landmarks = results.pose_landmarks.landmark

            # Right side landmarks
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Left side landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle at the elbows
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Calculate the percentage of rep completion for both arms
            # Assuming full extension is 170 degrees and bottom is 90 degrees
            right_rep_completion = np.interp(right_arm_angle, [90, 170], [0, 100])
            left_rep_completion = np.interp(left_arm_angle, [90, 170], [0, 100])

            # Form check for both arms
            if (right_arm_angle >= 75 and right_arm_angle <= 175) and (left_arm_angle >= 75 and left_arm_angle <= 175):
                form_feedback = "Good Form"
            else:
                form_feedback = "Bad Form: Adjust your posture"

            # Shoulder press counter logic for both arms
            if right_arm_angle > 160:
                stage["right"] = "down"
            if right_arm_angle < 90 and stage["right"] == 'down':
                stage["right"] = "up"

            if left_arm_angle > 160:
                stage["left"] = "down"
            if left_arm_angle < 90 and stage["left"] == 'down':
                stage["left"] = "up"

            # Count rep when both arms reach the full up position
            if stage["right"] == "up" and stage["left"] == "up":
                if stage["right"] == "up" and stage["left"] == "up":
                    counter += 1
                    print(f'Reps: {counter}')
                # Reset stages after counting a rep
                stage = {"left": None, "right": None}

            # Calculate the height of the progress bar based on completion percentage
            bar_height = int(max(right_rep_completion, left_rep_completion) * image.shape[0] / 130)

            # Define the coordinates for the progress bar (right side of the screen)
            x1 = image.shape[1] - 80  # X coordinate (keeping a margin from the right)
            y1 = image.shape[0] - bar_height  # Top Y coordinate (dependent on progress)
            x2 = image.shape[1] - 40  # Width of the bar (20 pixels wide)
            y2 = image.shape[1]  # Bottom Y coordinate (bottom of the screen)

            # Draw the progress bar (a green rectangle)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), -1)

            # Optional: Display the completion percentage text above the progress bar
            cv2.putText(image, f'{int(max(right_rep_completion, left_rep_completion))}%',
                        (x1 - 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw the progress bar on the right side of the screen
            bar_height = int(max(right_rep_completion,
                                 left_rep_completion) * 7.2)  # Scale the percentage to the window height (720px)
            cv2.rectangle(image, (1200, 720 - bar_height), (1260, 720), (0, 255, 0), -1)

            # Check if target reps are completed
            if counter >= target_reps:
                messagebox.showinfo("Workout Complete", "You have reached the target reps. Well done!")
                break

        except:
            pass

        # Draw the rep count on the image
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Count: {counter}',
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Display form feedback
        color = (0, 255, 0) if form_feedback == "Good Form" else (0, 0, 255)
        cv2.putText(image, form_feedback,
                    (320, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        # Show the image with landmarks and feedback
        cv2.imshow(window_name, image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
