import zmq
import threading
import torch
import requests
import io
import argparse
import csv
import unittest
from unittest.mock import patch, MagicMock
from torchvision import models, transforms
from PIL import Image
import json
import time
import os

# Usage
# First run one instance of the service
"""python zeromq_image_classification_service_with_threading.py"""
# Then one or more instances (supports threading) of
"""python zeromq_image_classification_service_with_threading.py --csv example_urls.csv"""

# Function to open the images and perform the predictions
def classify_image(url, model, transform, labels):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
        _, predicted = outputs.max(1)
        return labels[predicted.item()]
    except Exception as e:
        return f"Error processing {url}: {str(e)}"

# Split the urls and publish them
def process_urls(urls, model, transform, labels, pub_socket):
    for url in urls:
        result = classify_image(url, model, transform, labels)
        pub_socket.send_json({"url": url, "classification": result})

# Run the service and load the model (resnet50)
def image_processing_service():
    context = zmq.Context()
    rep_socket = context.socket(zmq.REP)
    rep_socket.bind("tcp://*:5557")

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5558")

    model = models.resnet50(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        # Default input size of resnet50
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # The following transformation values are the ones generally used with ImageNet datasets, on which resnet50 was also trained
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the imagenet pre-defined class labels
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(LABELS_URL).text.splitlines()

    print("Service started. Waiting for requests...")

    while True:
        # receive the list of URLs
        message = rep_socket.recv_json()
        urls = message.get("urls", [])
        rep_socket.send_json({"status": "processing"})

        # start a new thread to process the URLs. Using threading to be able to receive additional CSVs while processing the current one
        threading.Thread(
            target=process_urls,
            args=(urls, model, transform, labels, pub_socket),
            daemon=True
        ).start()

# run the cli service to accept CSVs with urls
def client_cli(csv_file):
    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect("tcp://localhost:5557")

    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5558")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    urls = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        urls = [row[0] for row in reader]

    req_socket.send_json({"urls": urls})
    response = req_socket.recv_json()
    print("Server response:", response)

    received_count = 0
    while received_count < len(urls):
        message = sub_socket.recv_json()
        print("Received:", message)
        received_count += 1

# simple unit tests
class TestImageClassification(unittest.TestCase):
    @patch("requests.get")
    def test_classify_image_mock(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = open("test_image.jpg", "rb").read()
        mock_get.return_value = mock_response

        from io import BytesIO
        test_image = Image.open(BytesIO(mock_response.content)).convert("RGB")
        self.assertIsNotNone(test_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV file containing image URLs", required=False)
    args = parser.parse_args()

    # If args, be ready to process a list of url's in csv. If not, just start the service
    if args.csv:
        client_cli(args.csv)
    else:
        image_processing_service()
        
        
        
# Dockerfile
DOCKERFILE_CONTENT = """
FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "zeromq_image_classification_service_with_threading.py"]
"""

with open("Dockerfile", "w") as f:
    f.write(DOCKERFILE_CONTENT)

print("Dockerfile created successfully.")

# Example CSV file for testing
EXAMPLE_CSV_CONTENT = """
https://www.princeton.edu/sites/default/files/styles/1x_full_2x_half_crop/public/images/2022/02/KOA_Nassau_2697x1517.jpg?itok=Bg2K7j7J
https://www.bellaandduke.com/wp-content/uploads/2024/10/pexels-%D0%B3%D0%B0%D0%BB%D0%B8%D0%BD%D0%B0-%D0%BB%D0%B0%D1%81%D0%B0%D0%B5%D0%B2%D0%B0-8522591-jpg-1024x678.webp
"""

with open("example_urls.csv", "w") as f:
    f.write(EXAMPLE_CSV_CONTENT.strip())

print("Example CSV file 'example_urls.csv' created successfully.")