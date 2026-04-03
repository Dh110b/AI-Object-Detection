from flask import Flask, request, jsonify  #importing files"""
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/detect": {"origins": "*"}})  #"""creating flask app instance"""

try:						#"""loading YOLO"""
    model = YOLO('yolov5su.pt')  # Or your custom model path
except Exception as e:
    print(f"Error loading model: {e}. Downloading...")
    model = YOLO('yolov5su')

model.to('cpu') # Force CPU usage if needed

def detect_objects(image):			#"""Performs object detection"""
    try:
        results = model(image)
        detections = []
        for *xyxy, conf, cls in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, xyxy)
            label = model.names[int(cls)]
            detections.append({"label": label, "bbox": [x1, y1, x2, y2]})
        return detections
    except Exception as e:
        print(f"Error in detection: {e}")
        return []  # Return empty list if detection fails

@app.route('/detect', methods=['POST'])		#"""ensures the image file is provided"""
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:				#"""Image Processing"""
        image_file = request.files['image']
        nparr = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error decoding image")
            return jsonify({'error': 'Error decoding image'}), 400

        results = detect_objects(image)		#"""running object Detection"""
        return jsonify(results)

    except Exception as e:			#"""Error Handling"""
        print(f"Error in /detect route: {e}")
        return jsonify({'error': 'An error occurred during detection'}), 500

if __name__ == '__main__':		#"""running the flask app"""
    app.run(debug=True)  # Set debug=False for production