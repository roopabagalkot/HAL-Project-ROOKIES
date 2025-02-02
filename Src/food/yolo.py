import cv2
import requests
from ultralytics import YOLO

# Flask server URL
FLASK_SERVER_URL = 'http://127.0.0.1:5000/update'

# Load your YOLO model with the trained weights
model_path = "bestV8.pt"  # Replace with the path to your weights file
yolo_model = YOLO(model_path)

# Start the webcam for real-time detection
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from the webcam.")
        break

    # Perform object detection on the frame
    results = yolo_model(frame)

    # Extract results for bounding boxes, class labels, and confidences
    boxes = results[0].boxes.xyxy  # Bounding box coordinates
    labels = results[0].boxes.cls  # Class IDs
    confidences = results[0].boxes.conf  # Confidence scores

    detected_objects = []

    # Draw the detections on the frame and store detected objects
    for box, label, conf in zip(boxes, labels, confidences):
        if conf > 0.6:  # Set a confidence threshold
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            class_name = yolo_model.names[int(label)]  # Get class name
            detected_objects.append(class_name)  # Store detected object

            # Draw the bounding box and label
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Send detected objects to the Flask server
    if detected_objects:
        try:
            response = requests.post(FLASK_SERVER_URL, json={'object_name': detected_objects})
            if response.status_code == 200:
                print(f"Sent detected objects: {detected_objects}")
            else:
                print(f"Failed to send data to Flask server. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Flask server: {e}")

    # Display the annotated frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
