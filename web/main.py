from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model
from utils.preprocess import preprocess_image
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import tensorflow as tf
import json
import difflib

app = FastAPI()

# Load disease descriptions from the JSON file
description_file_path = "web\\disease_des.json"
# Load JSON with UTF-8 encoding to avoid Windows decoding issues
with open(description_file_path, "r", encoding="utf-8") as f:
    disease_descriptions = json.load(f)

# Configure static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load model and class labels
model = load_model("models/skindisease.keras")
class_labels = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma', 'Nevus',
                'Pigmented Benign Keratosis', 'Dermatofibroma', 'Seborrheic Keratosis',
                'Squamous Cell Carcinoma', 'Vascular Lesion']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    try:
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process and predict
        processed_img = preprocess_image(image)
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        predicted_class = class_labels[np.argmax(prediction)]

        # Get description with fallback logic
        description_data = disease_descriptions.get(predicted_class)
        
        if not description_data:
            closest_match = difflib.get_close_matches(
                predicted_class, 
                disease_descriptions.keys(), 
                n=1, 
                cutoff=0.6
            )
            if closest_match:
                description_data = disease_descriptions[closest_match[0]]
                predicted_class = closest_match[0]
            else:
                description_data = {
                    "description": "No description available.",
                    "treatment": "No treatment information available."
                }

        return JSONResponse(content={
            "prediction": predicted_class,
            "description": description_data
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )