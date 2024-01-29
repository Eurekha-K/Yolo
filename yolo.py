import io
import base64
from fastapi import File, UploadFile, HTTPException,FastAPI
from starlette.responses import Response
from ultralytics import YOLO
import numpy as np
import cv2
# ... other imports
app=FastAPI()

@app.post("/detect/")

async def detect_objects(file: UploadFile):
    try:
        # Process the uploaded image
        model=YOLO('Car.pt')
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Perform object detection
        detections = model.predict(image)

        # Encode the image as a base64 string
        image_encoded = base64.b64encode(image_bytes).decode("utf-8")

        # Format the JSON response with detection details
        response_json = {
            "detections": detections.to_list(),  # Convert YOLO's output to a list
            "image_base64": image_encoded
        }

        return Response(content=response_json, media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
