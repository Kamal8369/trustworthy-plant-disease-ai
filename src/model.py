import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes: int, dropout: float = 0.5):
    """Build the current frozen-backbone ResNet18 baseline without hardcoded class count."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes),
    )
    return model
