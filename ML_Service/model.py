# In your main.py
from transformers import pipeline

# Load a model that has ALREADY been trained on deepfake datasets
# This downloads the architecture AND the trained weights automatically
deepfake_detector = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model")

@app.post("/detect")
async def detect_media(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Run the image through the pre-trained model
    results = deepfake_detector(img)
    
    return {"verdict": results}