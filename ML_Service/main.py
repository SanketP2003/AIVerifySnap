import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import pipeline

# Initialize the FastAPI application
app = FastAPI(
    title="AIVerifySnap AI Service Layer",
    description="Deepfake detection microservice using a pre-trained Vision Transformer.",
    version="2.0.0"
)

# Load the pre-trained model
# Note: The first time you run this script, it will take a minute or two 
# to download the model weights (~several hundred megabytes) from Hugging Face.
print("Loading pre-trained deepfake detector...")
try:
    deepfake_detector = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/")
async def health_check():
    """Simple endpoint to verify the service is running."""
    return {"status": "healthy", "service": "ai-detection-layer", "model": "pre-trained"}

@app.post("/detect")
async def detect_media(file: UploadFile = File(...)):
    """Accepts an image upload and returns a deepfake detection verdict."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # 1. Read the uploaded image into memory
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Run the image through the pre-trained Hugging Face model
        # The pipeline usually returns a list of dictionaries, e.g.:
        # [{'label': 'Fake', 'score': 0.98}, {'label': 'Real', 'score': 0.02}]
        results = deepfake_detector(img)
        
        # 3. Extract the highest confidence result
        # Sort by score in descending order to get the top prediction
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_prediction = sorted_results[0]
        
        # 4. Format the final response cleanly
        return {
            "filename": file.filename,
            "verdict": top_prediction['label'],
            "confidence": round(top_prediction['score'] * 100, 2),
            "raw_output": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    # Run the server locally on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)