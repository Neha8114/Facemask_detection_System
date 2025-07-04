import requests

# Custom Vision Prediction endpoint
url = "https://facemask-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/b55e36e1-65e0-47d4-86d6-6b99b7943f78/classify/iterations/Iteration2/image"

# Prediction Key
prediction_key = "DocDkGjJb430yqqpJiLCTbbQq8qBhN7BzJpkYd7lGIyAZHdnwSzIJQQJ99BGACYeBjFXJ3w3AAAIACOGMUbI"

image_path = "test_image2.jpg"

headers = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/octet-stream"
}

with open(image_path, "rb") as img_file:
    image_data = img_file.read()

response = requests.post(url, headers=headers, data=image_data)

if response.status_code == 200:
    predictions = response.json()["predictions"]
    for pred in predictions:
        tag = pred["tagName"]
        prob = pred["probability"]
        print(f"Prediction: {tag} ({prob*100:.2f}%)")
else:
    print("Error:", response.status_code)
    print(response.text)
