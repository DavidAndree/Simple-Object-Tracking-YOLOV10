from ultralytics import YOLO 
import cvzone
import cv2

""" YOLO MODEL """
model = YOLO('yolov10n.pt')

# Access webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Checks webcam connectivity
while True:
    ret, image = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Run YOLO model 
    results = model(image)

    # Process each detected box
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            # Get confidence score
            confidence = box.conf[0].numpy() * 100  # in percentage
            # Get detected class
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(
                image, 
                f'{class_detected_name} {confidence:.2f}%', 
                (x1 + 8, y1 - 12), 
                thickness=2, 
                scale=1.5, 
                colorR=(255, 255, 255),
                colorT=(0, 0, 255)
            )

    # Display the resulting frame
    cv2.imshow('frame', image)
    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release  program
cap.release()
cv2.destroyAllWindows()