import cv2

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image (noise remove)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold (object highlight)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    # Find contours (object detection)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Small noise ignore madbeku
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()