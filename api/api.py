import numpy as np
import torch
import cv2
import mlflow
from fastapi import FastAPI, UploadFile
from src.transform import val_transform

app = FastAPI()


model = mlflow.pytorch.load_model("models:/PlantDisease@champion")

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

CLASS_NAMES = {
    "0": "Pepper__bell___Bacterial_spot",
    "1": "Pepper__bell___healthy",
    "2": "Potato___Early_blight",
    "3": "Potato___Late_blight",
    "4": "Potato___healthy",
    "5": "Tomato_Bacterial_spot",
    "6": "Tomato_Early_blight",
    "7": "Tomato_Late_blight",
    "8": "Tomato_Leaf_Mold",
    "9": "Tomato_Septoria_leaf_spot",
    "10": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "11": "Tomato__Target_Spot",
    "12": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "13": "Tomato__Tomato_mosaic_virus",
    "14": "Tomato_healthy",
}


@app.post("/predict")
async def predict(file: UploadFile):
    # Read and decode image
    image_bytes = await file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = val_transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "class": CLASS_NAMES[str(predicted.item())],
        "confidence": round(confidence.item(), 4),
    }
