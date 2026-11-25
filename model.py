"""
Model architectures for galaxy morphology classification.

  resnet18         early fusion  - the 3 bands enter one ResNet-18 as a 3-channel image
  efficientnet_b0  early fusion  - same input, different architecture family
  multiband        late fusion   - one ResNet-18 stream per band (IR / VIS / UV); the three feature
  #vectors are concatenated and classified together (true multi-modal)
  resnet50, simple - optional extras
"""

import torch
import torch.nn as nn
from torchvision import models

ARCHS = ["resnet18", "efficientnet_b0", "multiband", "resnet50", "simple"]
DEFAULT_ARCHS = ["resnet18", "multiband", "efficientnet_b0"]
BAND_NAMES = ["IR", "VIS", "UV"]

DESCRIPTIONS = {
    "resnet18": "ResNet-18 on the fused IR/VIS/UV composite (early fusion)",
    "efficientnet_b0": "EfficientNet-B0 on the fused composite (early fusion)",
    "multiband": "3 ResNet-18 streams, one per band, features concatenated (late fusion)",
    "resnet50": "ResNet-50 on the fused composite",
    "simple": "Small 3-layer CNN baseline",
}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
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
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _resnet18_band_stream(pretrained=True):
    """ResNet-18 trunk for ONE band: single-channel input, 512-d output.
    The pretrained RGB filters of conv1 are summed so ImageNet knowledge is kept."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    old = m.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
    m.conv1 = new
    m.fc = nn.Identity()
    return m


class MultiBandNet(nn.Module):
    """
    Late-fusion multi-modal network. Input is the usual (B, 3, H, W) tensor whose channels are
    IR, VIS, UV. Every band goes through its own ResNet-18 stream; the 3 x 512 features are
    concatenated and passed to an MLP head.

    Band dropout (training only) randomly silences a whole band's features so the network cannot rely
    on a single wavelength and still works when one band is weak.
    """

    def __init__(self, num_classes=3, pretrained=True, band_dropout=0.15):
        super().__init__()
        self.streams = nn.ModuleList([_resnet18_band_stream(pretrained) for _ in BAND_NAMES])
        self.band_dropout = band_dropout
        self.head = nn.Sequential(
            nn.Linear(512 * len(BAND_NAMES), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def embed(self, x):
        feats = [stream(x[:, i:i + 1]) for i, stream in enumerate(self.streams)]
        if self.training and self.band_dropout > 0:
            keep = (torch.rand(x.size(0), len(feats), device=x.device) > self.band_dropout).float()
            keep[keep.sum(1) == 0] = 1.0                      # never silence all bands
            feats = [f * keep[:, i:i + 1] for i, f in enumerate(feats)]
        return torch.cat(feats, dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


def get_model(arch="resnet18", num_classes=3, pretrained=True):
    arch = (arch or "resnet18").lower()
    if arch == "simple":
        return SimpleCNN(num_classes=num_classes)
    if arch == "multiband":
        return MultiBandNet(num_classes=num_classes, pretrained=pretrained)
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, num_classes))
        return model
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
        return model
    if arch != "resnet18":
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHS}")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def extract_features(model, x):
    """Penultimate-layer embedding (B, D) used for the PCA / t-SNE scatter plots."""
    if isinstance(model, MultiBandNet):
        return model.embed(x)
    if isinstance(model, SimpleCNN):
        return model.features(x).flatten(1)
    if isinstance(model, models.ResNet):
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        return torch.flatten(model.avgpool(x), 1)
    if isinstance(model, models.EfficientNet):
        return torch.flatten(model.avgpool(model.features(x)), 1)
    raise TypeError(f"No feature extractor for {type(model).__name__}")
