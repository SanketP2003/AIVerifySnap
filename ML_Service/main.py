import io
import base64
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms

from model import AIVerifySnapModel, AIVerifySnapModelV1
from utils import generate_ela

app = FastAPI(
    title="AIVerifySnap AI Service Layer",
    description="Deepfake detection microservice using dual-stream ResNet18 + ELA CNN fusion.",
    version="5.0.0",
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(__file__).parent / "artifacts" / "best_model.pt"

# These will be populated by load_model()
custom_model = None
custom_transforms_rgb = None
custom_transforms_ela = None
custom_image_size = None
custom_class_mapping = None  # e.g. {'Fake': 0, 'Real': 1}

# HuggingFace fallback (only used when custom checkpoint is missing)
hf_processor = None
hf_model = None
HF_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
HF_ID2LABEL = {0: "Fake", 1: "Real"}

using_custom_model = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _build_eval_transforms(image_size: int):
    """Build the same eval transforms used during training."""
    rgb = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ela = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    return rgb, ela


def load_model() -> None:
    """Load the custom trained model, falling back to HuggingFace if unavailable."""
    global custom_model, custom_transforms_rgb, custom_transforms_ela
    global custom_image_size, custom_class_mapping, using_custom_model
    global hf_processor, hf_model

    # --- Try custom checkpoint first ---
    if CHECKPOINT_PATH.exists():
        print(f"Loading custom model from {CHECKPOINT_PATH} ...")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            custom_image_size = ckpt.get("image_size", 224)
            custom_class_mapping = ckpt.get("class_mapping", {"Fake": 0, "Real": 1})
            state_dict = ckpt["model_state_dict"]

            # Try the V1 architecture first (matches the trained checkpoint),
            # then fall back to the latest architecture.
            model = None
            for model_cls, name in [(AIVerifySnapModelV1, "V1"), (AIVerifySnapModel, "latest")]:
                try:
                    candidate = model_cls(freeze_backbone=True)
                    candidate.load_state_dict(state_dict)
                    candidate.eval()
                    model = candidate
                    print(f"Loaded checkpoint with {name} architecture ({model_cls.__name__}).")
                    break
                except RuntimeError as e:
                    print(f"{name} architecture ({model_cls.__name__}) failed: {e}")

            if model is not None:
                custom_model = model
                custom_transforms_rgb, custom_transforms_ela = _build_eval_transforms(custom_image_size)
                using_custom_model = True
                best_acc = ckpt.get("best_val_acc", "N/A")
                print(f"Custom model loaded! image_size={custom_image_size}, "
                      f"class_mapping={custom_class_mapping}, best_val_acc={best_acc}")
                return
            else:
                print("Could not load checkpoint with any known architecture.")
        except Exception as e:
            print(f"Failed to load custom checkpoint: {e}")
            custom_model = None

    # --- Fallback: HuggingFace SigLIP model ---
    print(f"Falling back to HuggingFace model: {HF_MODEL_NAME}")
    try:
        from transformers import AutoImageProcessor, SiglipForImageClassification
        hf_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
        hf_model = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
        hf_model.eval()
        using_custom_model = False
        print("HuggingFace fallback model loaded successfully.")
    except Exception as e:
        print(f"FATAL - Failed to load fallback model: {e}")


load_model()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _infer_custom(img: Image.Image):
    """Run inference with the custom dual-stream AIVerifySnapModel."""
    rgb_tensor = custom_transforms_rgb(img).unsqueeze(0)          # (1, 3, H, W)
    ela_img = generate_ela(img, quality=90)
    ela_tensor = custom_transforms_ela(ela_img).unsqueeze(0)      # (1, 3, H, W)

    with torch.no_grad():
        logit = custom_model(rgb_tensor, ela_tensor)              # (1, 1)
        prob_real = torch.sigmoid(logit).item()

    # class_mapping: {'Fake': 0, 'Real': 1}
    # sigmoid > 0.5 → Real (label 1), sigmoid ≤ 0.5 → Fake (label 0)
    prob_fake = 1.0 - prob_real

    if prob_real >= 0.5:
        verdict = "Real"
        confidence = prob_real
    else:
        verdict = "Fake"
        confidence = prob_fake

    scores = [
        {"label": "Fake", "score": round(prob_fake, 6)},
        {"label": "Real", "score": round(prob_real, 6)},
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)

    return verdict, round(confidence * 100, 2), scores


def _infer_huggingface(img: Image.Image):
    """Run inference with the HuggingFace SigLIP fallback model."""
    inputs = hf_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    scores = [{"label": HF_ID2LABEL[idx], "score": round(prob, 6)}
              for idx, prob in enumerate(probs)]
    scores.sort(key=lambda x: x["score"], reverse=True)
    top = scores[0]

    return top["label"], round(top["score"] * 100, 2), scores


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def health_check():
    model_loaded = custom_model is not None or hf_model is not None
    model_desc = ("AIVerifySnapModel (dual-stream RGB+ELA)"
                  if using_custom_model
                  else f"HuggingFace SigLIP fallback ({HF_MODEL_NAME})")
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "service": "ai-detection-layer",
        "model": model_desc,
        "using_custom_model": using_custom_model,
    }


@app.post("/detect")
async def detect_media(file: UploadFile = File(...)):
    if custom_model is None and hf_model is None:
        raise HTTPException(status_code=503, detail="No model is loaded. Check server logs.")

    try:
        start_time = time.time()

        image_bytes = await file.read()

        # Try to open as image — don't rely solely on content_type header
        # because the Spring Boot backend proxy may send application/octet-stream
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

        # --- Run inference ---
        if using_custom_model and custom_model is not None:
            verdict, confidence, scores = _infer_custom(img)
        else:
            verdict, confidence, scores = _infer_huggingface(img)

        # --- Generate ELA visualisation ---
        max_ela_dim = 1024
        img_for_ela = img.copy()
        if max(img_for_ela.size) > max_ela_dim:
            img_for_ela.thumbnail((max_ela_dim, max_ela_dim), Image.LANCZOS)
        ela_img = generate_ela(img_for_ela, quality=90)

        ela_array = np.array(ela_img, dtype=np.float32)
        ela_mean = float(np.mean(ela_array))
        ela_std = float(np.std(ela_array))
        ela_max = float(np.max(ela_array))

        ela_buffer = io.BytesIO()
        ela_img.save(ela_buffer, format="PNG")
        ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode("utf-8")

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "filename": file.filename,
            "verdict": verdict,
            "confidence": confidence,
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
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)