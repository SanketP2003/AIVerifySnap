# AIVerifySnap Model Weights Directory

This directory is intended to store trained model weights for the AIVerifyNet deepfake detection model.

## Supported Files

- `aiverifynet.pth` - PyTorch state_dict for AIVerifyNet dual-stream model
- `model.pt` - TorchScript compiled model (alternative format)

## Training Your Own Model

To train the AIVerifyNet model, you'll need:

1. **Dataset**: A labeled dataset of real and fake images (e.g., FaceForensics++, DFDC)
2. **Training Script**: Use the architecture defined in `app/aiverifynet.py`

### Example Training Code

```python
import torch
from app.aiverifynet import AIVerifyNet

# Initialize model
model = AIVerifyNet(pretrained=True, dropout_rate=0.3)

# Training loop (pseudo-code)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for rgb_batch, ela_batch, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(rgb_batch, ela_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save trained weights
torch.save(model.state_dict(), 'models/aiverifynet.pth')
```

## Mock Mode

If no trained weights are available, the service will run in "mock mode" using:
- Untrained network weights (random predictions)
- Heuristic-based ELA analysis as fallback

Set `ALLOW_UNTRAINED=1` in environment variables to enable mock mode.

