"""Model definitions: pretrained torchvision CNNs with a new classification head."""
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(name, num_classes, pretrained=True):
    if name != "efficientnet_b0":
        raise ValueError(f"Unsupported model: {name}")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    # Replace the 1000-class ImageNet head with one output per skin condition
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
