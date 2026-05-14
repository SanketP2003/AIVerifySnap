import torch
import torch.nn as nn
from torchvision import models


class AIVerifySnapModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.rgb_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.backbone_trainable = not freeze_backbone
        if freeze_backbone:
            for name, param in self.resnet.named_parameters():
                if not name.startswith("layer4"):
                    param.requires_grad = False

        self.ela_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, rgb, ela):
        if self.backbone_trainable:
            rgb_features = self.resnet(rgb)
        else:
            with torch.no_grad():
                rgb_raw = self.resnet(rgb)
            rgb_features = rgb_raw.detach()
        rgb_features = self.rgb_head(rgb_features)
        ela_features = self.ela_cnn(ela)

        combined = torch.cat((rgb_features, ela_features), dim=1)
        return self.classifier(combined)


class AIVerifySnapModelV1(nn.Module):
    """Architecture matching the trained checkpoint (best_model.pt).

    Differences from the latest AIVerifySnapModel:
    - resnet.fc is a Linear(512, 128) instead of Identity() + separate rgb_head
    - classifier is a simpler 2-layer network: Linear(384, 128) -> Linear(128, 1)
      (384 = 128 from resnet.fc + 256 from ela_cnn)
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features  # 512
        # The trained checkpoint used a Linear head in the resnet, not Identity
        self.resnet.fc = nn.Linear(in_features, 128)

        self.backbone_trainable = not freeze_backbone
        if freeze_backbone:
            for name, param in self.resnet.named_parameters():
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False

        self.ela_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # 128 (resnet.fc) + 256 (ela_cnn) = 384
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, ela):
        if self.backbone_trainable:
            rgb_features = self.resnet(rgb)
        else:
            with torch.no_grad():
                rgb_raw = self.resnet(rgb)
            rgb_features = rgb_raw.detach()
        ela_features = self.ela_cnn(ela)

        combined = torch.cat((rgb_features, ela_features), dim=1)
        return self.classifier(combined)