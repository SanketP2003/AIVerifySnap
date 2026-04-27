"""Compare custom model vs HuggingFace model accuracy on validation data."""
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, SiglipForImageClassification
from model import AIVerifySnapModelV1
from utils import generate_ela

# ---- Load custom model ----
ckpt = torch.load("artifacts/best_model.pt", map_location="cpu", weights_only=False)
custom_model = AIVerifySnapModelV1(freeze_backbone=True)
custom_model.load_state_dict(ckpt["model_state_dict"])
custom_model.eval()
img_size = ckpt["image_size"]  # 160

eval_rgb = transforms.Compose([
    transforms.Resize(img_size + 32),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_ela = transforms.Compose([
    transforms.Resize(img_size + 32),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

# ---- Load HuggingFace model ----
HF_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
hf_proc = AutoImageProcessor.from_pretrained(HF_NAME)
hf_model = SiglipForImageClassification.from_pretrained(HF_NAME)
hf_model.eval()
hf_id2label = {0: "Fake", 1: "Real"}

def predict_custom(img):
    rgb = eval_rgb(img).unsqueeze(0)
    ela = eval_ela(generate_ela(img, quality=90)).unsqueeze(0)
    with torch.no_grad():
        logit = custom_model(rgb, ela)
        p = torch.sigmoid(logit).item()
    return ("Real" if p >= 0.5 else "Fake"), (p if p >= 0.5 else 1-p)

def predict_hf(img):
    inputs = hf_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = hf_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    idx = 0 if probs[0] > probs[1] else 1
    return hf_id2label[idx], probs[idx]

# ---- Test on validation ----
n = 25
val_fake = sorted(os.listdir("Dataset/Validation/Fake"))[:n]
val_real = sorted(os.listdir("Dataset/Validation/Real"))[:n]

custom_correct = 0
hf_correct = 0
total = 0

for label, files, d in [("Fake", val_fake, "Dataset/Validation/Fake"), ("Real", val_real, "Dataset/Validation/Real")]:
    print(f"\n--- {label} ({len(files)} samples) ---")
    for f in files:
        img = Image.open(os.path.join(d, f)).convert("RGB")
        cv, cc = predict_custom(img)
        hv, hc = predict_hf(img)
        c_ok = cv == label
        h_ok = hv == label
        if c_ok: custom_correct += 1
        if h_ok: hf_correct += 1
        total += 1
        c_mark = "OK" if c_ok else "XX"
        h_mark = "OK" if h_ok else "XX"
        print(f"  {f}: Custom=[{c_mark}] {cv}({cc*100:.1f}%)  HF=[{h_mark}] {hv}({hc*100:.1f}%)")

print(f"\n{'='*50}")
print(f"Custom model accuracy: {custom_correct}/{total} = {custom_correct/total*100:.1f}%")
print(f"HuggingFace accuracy:  {hf_correct}/{total} = {hf_correct/total*100:.1f}%")
