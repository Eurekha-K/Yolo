import pandas as pd
from pydantic import BaseModel
import numpy as np
# Import necessary modules


import os

import shutil
import random

import numpy as np
import pandas as pd



from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel

from collections import Counter
import numpy as np
from io import BytesIO
import cv2




from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os


from fastapi import FastAPI
from pydantic import BaseModel
app=FastAPI()


# Make predictions on an image
@app.post("/car-detect/")
async def detect_objects(file: UploadFile):
 model = YOLO('Car.pt')
 # Process the uploaded image for object detection
 image_bytes = await file.read()
 image = np.frombuffer(image_bytes, dtype=np.uint8)
 image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
 # Perform object detection with YOLOv8
 results = model(image)
 
 
 print('The actual class is Acura Integra Type R 2001')
# Display the predicted class with the highest probability
 names_dict = results[0].names
 probs = results[0].probs.data.tolist()
 return {'detection':names_dict[np.argmax(probs)]}


 
