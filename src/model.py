from torchvision import models
import torch.nn as nn


class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=15, freeze_backbone=True):
        super().__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

    def forward(self, x):
        return self.model(x)
