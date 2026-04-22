import torch
import torch.nn as nn
from torchvision import models

class AIVerifySnapModel(nn.Module):
    def __init__(self):
        super(AIVerifySnapModel, self).__init__()
        
        # Stream 1: Spatial/RGB features using a pre-trained ResNet50
        # Replaces the final classification layer to output a 256-d feature vector
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        
        # Stream 2: ELA features using a custom lightweight CNN
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
        
        # Fusion Layer: Concatenates both streams and classifies
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, ela):
        rgb_features = self.resnet(rgb)
        ela_features = self.ela_cnn(ela)
        
        # Concatenate features from both streams
        combined = torch.cat((rgb_features, ela_features), dim=1)
        
        # Returns logits; apply sigmoid during evaluation/inference when needed.
        return self.classifier(combined)