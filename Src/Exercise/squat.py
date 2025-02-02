import cv2
import numpy as np
import time
from pose_detector import PoseDetector  # Ensure pose_detector.py is in the same directory
import tkinter as tk
from tkinter import messagebox

def track_squat(cap, detector, target_reps, root):
    count = 0
    dir = 0  # 0: down, 1: up
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Resize the image for better performance
        img = cv2.resize(img, (1280, 720))

        # Detect the pose
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Calculate angle of the left leg
            angle = detector.findAngle(img, 23, 25, 27)  # Left Leg

            # Squat detection range
            low = 70
            high = 160
            per = np.interp(angle, (low, high), (0, 100))
            bar = np.interp(angle, (low, high), (650, 100))

            # Check for squat repetitions
            if per == 100:  # Fully up
                if dir == 1:  # Coming up from a squat
                    count += 1
                    dir = 0  # Set direction to up

            if per == 0:  # Fully down
                if dir == 0:  # Going down
                    dir = 1  # Set direction to down

            # Draw bar
            cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

            # Display squat count
            cv2.putText(img, f'Squats: {int(count)} / {target_reps}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Display FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Check if the target repetitions are completed
            if count >= target_reps:
                messagebox.showinfo("Workout Completed", "Great job, workout completed!")
                break

        # Display the image
        cv2.imshow("Squat Tracker", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()  # Close the Tkinter window after the workout is completed

def start_tracking(root, entry, cap, detector):
    target_reps = int(entry.get())
    root.withdraw()  # Hide the Tkinter window while tracking
    track_squat(cap, detector, target_reps, root)

if __name__ == "__main__":
    # Initialize Tkinter
    root = tk.Tk()
    root.title("Squat Tracker")

    # Create and place widgets in Tkinter window
    tk.Label(root, text="Enter the number of squats you want to complete:").pack(pady=10)
    entry = tk.Entry(root)
    entry.pack(pady=10)

    # Start button to begin tracking
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    detector = PoseDetector()
    start_button = tk.Button(root, text="Start Workout", command=lambda: start_tracking(root, entry, cap, detector))
    start_button.pack(pady=10)

    # Run the Tkinter main loop
    root.mainloop()
