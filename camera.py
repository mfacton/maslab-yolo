from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("runs/detect/train9/weights/best.pt")
model.to("cuda")

# Open the camera (use 0 for the default webcam, or 1 for an external USB camera)
camera_index = 2  # Change to 0 if you're using the default webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}")
    exit()

# Loop to capture and process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        break

    # Run YOLO on the current frame
    results = model(frame)

    # Visualize the detections directly on the frame
    annotated_frame = results[0].plot()  # YOLO's built-in visualization

    # Display the annotated frame
    cv2.imshow("YOLO Live Output", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

