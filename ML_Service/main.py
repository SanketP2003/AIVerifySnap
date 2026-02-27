import io
import base64
import time
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

from utils import generate_ela

# Initialize the FastAPI application
app = FastAPI(
    title="AIVerifySnap AI Service Layer",
    description="Deepfake detection microservice using a pre-trained SigLIP Vision Transformer.",
    version="3.0.0"
)

# ------------------------------------------------------------------
# Model Loading — using the OFFICIAL inference approach from the model card
# Model: prithivMLmods/Deep-Fake-Detector-Model  (SigLIP-based)
# Label space: Class 0 = "Fake", Class 1 = "Real"
# ------------------------------------------------------------------
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"

# Explicit label mapping from the model card
id2label = {0: "Fake", 1: "Real"}

print(f"Loading model: {MODEL_NAME} ...")
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()  # set to evaluation mode
    print("Model and processor loaded successfully!")
except Exception as e:
    processor = None
    model = None
    print(f"FATAL — Error loading model: {e}")


@app.get("/")
async def health_check():
    """Simple endpoint to verify the service is running."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "service": "ai-detection-layer",
        "model": MODEL_NAME,
    }


@app.post("/detect")
async def detect_media(file: UploadFile = File(...)):
    """Accepts an image upload and returns a deepfake detection verdict with ELA analysis."""

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        start_time = time.time()

        # 1. Read the uploaded image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Preprocess with the OFFICIAL AutoImageProcessor (correct resize/norm for SigLIP)
        inputs = processor(images=img, return_tensors="pt")

        # 3. Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        # 4. Build per-class scores
        scores = []
        for idx, prob in enumerate(probs):
            scores.append({
                "label": id2label[idx],
                "score": round(prob, 6)
            })

        # Sort by score descending — top prediction first
        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[0]

        # 5. Generate real ELA (Error Level Analysis)
        ela_img = generate_ela(img, quality=90)

        # Compute ELA statistics
        ela_array = np.array(ela_img, dtype=np.float32)
        ela_mean = float(np.mean(ela_array))
        ela_std = float(np.std(ela_array))
        ela_max = float(np.max(ela_array))

        # Encode ELA image as base64 PNG for the frontend
        ela_buffer = io.BytesIO()
        ela_img.save(ela_buffer, format="PNG")
        ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode("utf-8")

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        # 6. Build the response
        return {
            "filename": file.filename,
            "verdict": top["label"],
            "confidence": round(top["score"] * 100, 2),
            "raw_output": scores,
            "ela": {
                "mean": round(ela_mean, 4),
                "std": round(ela_std, 4),
                "max": round(ela_max, 4),
                "image_base64": ela_base64,
            },
            "processing_time_ms": elapsed_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    # Run the server locally on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)