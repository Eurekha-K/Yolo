from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
from ultralytics import YOLO
from random import randint
import uuid
import numpy as np
import cv2

#upload/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
app = FastAPI()
origins=["*"]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
IMAGEDIR = "images/"
 

 
 
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    model=YOLO('Car.pt')
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    image = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
 # Perform object detection with YOLOv8
    results = model(image)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    return {"response":names_dict[np.argmax(probs)]}
 
 
@app.get("/show/")
async def read_random_file():
 
    # get random file from the image directory
    files = os.listdir(IMAGEDIR)
    print(list)
    random_index = 0
 
    path = f"{IMAGEDIR}{files[random_index]}"
     
    return FileResponse(path)