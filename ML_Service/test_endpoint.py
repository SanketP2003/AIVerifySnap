import requests
import json
import os
import sys

# Test against ML service — we'll temporarily switch to HuggingFace
# by renaming the checkpoint
ckpt = "artifacts/best_model.pt"
ckpt_bak = "artifacts/best_model.pt.bak"

if "--hf" in sys.argv:
    # Rename checkpoint to force HuggingFace fallback
    if os.path.exists(ckpt):
        os.rename(ckpt, ckpt_bak)
        print("Renamed checkpoint to force HuggingFace fallback")
        print("Restart ML service and run this script again without --hf")
    else:
        print("Checkpoint already renamed")
elif "--restore" in sys.argv:
    if os.path.exists(ckpt_bak):
        os.rename(ckpt_bak, ckpt)
        print("Restored checkpoint")
    else:
        print("No backup to restore")
else:
    # Check health
    health = requests.get("http://localhost:8000/").json()
    print("Model:", health.get("model"))
    print("Using custom:", health.get("using_custom_model"))
    print()
    
    def test_image(path, expected):
        with open(path, "rb") as f:
            resp = requests.post("http://localhost:8000/detect", 
                               files={"file": (os.path.basename(path), f, "image/jpeg")})
        d = resp.json()
        ok = d["verdict"] == expected
        return ok, d["verdict"], d["confidence"]
    
    val_fake = "Dataset/Validation/Fake"
    val_real = "Dataset/Validation/Real"
    
    # Test 20 from each class in validation
    n = 20
    fake_files = sorted(os.listdir(val_fake))[:n]
    real_files = sorted(os.listdir(val_real))[:n]
    
    correct = 0
    total = 0
    
    print(f"--- VAL FAKE ({n} samples) ---")
    for f in fake_files:
        ok, v, c = test_image(os.path.join(val_fake, f), "Fake")
        mark = "OK" if ok else "WRONG"
        print(f"  [{mark}] {f}: {v} ({c}%)")
        if ok: correct += 1
        total += 1
    
    print(f"\n--- VAL REAL ({n} samples) ---")
    for f in real_files:
        ok, v, c = test_image(os.path.join(val_real, f), "Real")
        mark = "OK" if ok else "WRONG"
        print(f"  [{mark}] {f}: {v} ({c}%)")
        if ok: correct += 1
        total += 1
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
