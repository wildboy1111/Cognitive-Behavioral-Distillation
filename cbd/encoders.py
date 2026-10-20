"""Image, EEG and eye-tracking encoders shared by all experiments."""

import torch
from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    """VGG16 with 256-dimensional global/local image features."""

    def __init__(self, n_classes=5, pretrained=True):
        d = 256
        super().__init__()
        self.hidden_dim = d
        self.n_classes = n_classes
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = vgg.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        for idx in [26, 28]:
            for param in self.backbone[idx].parameters():
                param.requires_grad = True
        c = 512
        self.global_block = nn.Sequential(
            nn.Conv2d(c, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(784, self.hidden_dim),
        )
        self.local_block = nn.Sequential(
            nn.Conv2d(c, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.Flatten(),
        )
        self.fusion_block = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        self.classifier = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        g = self.global_block(feat)
        l = self.local_block(feat)
        v = self.fusion_block(torch.concat([g, l], dim=1))
        logits = self.classifier(v)
        return {"logits": logits, "global_feat, local_feat": (g, l)}


class EEGEncoder(nn.Module):
    """Temporal/spatial convolutions with a shape-aware projection layer."""

    def __init__(
        self,
        n_classes=3,
        channels=62,
        samples=250,
        dropout=0.5,
        temporal_kernel=64,
        separable_kernel=16,
        temporal_filters=8,
        depth_multiplier=2,
        output_filters=16,
    ):
        super().__init__()
        self.temporal_filters = temporal_filters
        self.output_filters = output_filters
        self.depth_multiplier = depth_multiplier
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.temporal_kernel = temporal_kernel
        self.separable_kernel = separable_kernel
        self.drop_out = dropout
        block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.temporal_filters,
                kernel_size=(1, self.temporal_kernel),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.temporal_filters),
        )
        block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.temporal_filters,
                out_channels=self.temporal_filters * self.depth_multiplier,
                kernel_size=(self.channels, 1),
                groups=self.temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.temporal_filters * self.depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=self.drop_out),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.depth_multiplier * self.temporal_filters,
                out_channels=self.depth_multiplier * self.temporal_filters,
                kernel_size=(1, separable_kernel),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.Conv2d(
                in_channels=self.depth_multiplier * self.temporal_filters,
                out_channels=self.output_filters,
                kernel_size=(1, 1),
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.output_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=self.drop_out),
        )
        # Keep the original parameter paths for strict legacy weight loading.
        self.EEGNetLayer = nn.Sequential(block1, block2, self.block3)
        self.project = nn.Sequential(
            nn.Linear(
                in_features=output_filters * (samples // 4 // 8),
                out_features=256,
                bias=True,
            )
        )
        self.ClassifierBlock = nn.Sequential(
            nn.Linear(in_features=256, out_features=self.n_classes, bias=True)
        )

    def forward(self, x, label=None):
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
        x = self.EEGNetLayer(x)
        x = x.view(x.size()[0], -1)
        g_feat = self.project(x)
        logits = self.ClassifierBlock(g_feat)
        return {"logits": logits, "global_feat": g_feat}


class EyeTrackingEncoder(nn.Module):
    """Time/feature convolutions followed by a fixed 256-dimensional pooling."""

    def __init__(
        self,
        n_classes=5,
        input_size=(250, 4),
        dropout=0.5,
        d=256,
    ):
        super(EyeTrackingEncoder, self).__init__()
        self.dropout = dropout
        self.hidden_dim = d
        self.local_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(input_size[0] // 4, 1), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, input_size[1])),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(4, 1)),
            nn.Dropout(self.dropout),
            nn.Conv2d(16, 8, (1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(d, n_classes)

    def forward(self, x, labels=None):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        l = self.local_block(x)
        logits = self.classifier(l)
        return {"logits": logits, "local_feat": l}
