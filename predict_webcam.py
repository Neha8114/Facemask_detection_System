import cv2
import requests
import numpy as np



# Prediction Key
PREDICTION_KEY = "DocDkGjJb430yqqpJiLCTbbQq8qBhN7BzJpkYd7lGIyAZHdnwSzIJQQJ99BGACYeBjFXJ3w3AAAIACOGMUbI"

# Custom Vision Prediction endpoint
ENDPOINT_URL = "https://facemask-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/b55e36e1-65e0-47d4-86d6-6b99b7943f78/classify/iterations/Iteration2/image"


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame as image
    image_path = "temp.jpg"
    cv2.imwrite(image_path, frame)

    # Read the image in binary
    with open(image_path, "rb") as f:
        image_data = f.read()


    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(ENDPOINT_URL, headers=headers, data=image_data)
    result = response.json()

    # Extract top prediction
    predictions = result.get("predictions", [])
    if predictions:
        top_pred = predictions[0]
        tag = top_pred['tagName']
        prob = top_pred['probability'] * 100
        label = f"{tag} ({prob:.2f}%)"

        # Set color
        color = (0, 255, 0) if tag.lower() == "with_mask" else (0, 0, 255)

        # Draw label on frame
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 3)

    # Show the frame
    cv2.imshow("Azure Mask Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
