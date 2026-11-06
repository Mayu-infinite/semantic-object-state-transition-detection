import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class RGBStateTransitionModel(nn.Module):
    def __init__(
        self,
        num_classes=4,
        hidden_dim=256,
        pretrained=True,
        freeze_encoder=False,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )
        self.feature_dim = 512

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.temporal_model = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, video):
        batch_size, time_steps, channels, height, width = video.shape

        frames = video.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        features = self.encoder(frames).flatten(1)

        features = features.reshape(
            batch_size,
            time_steps,
            self.feature_dim,
        )

        temporal_features, _ = self.temporal_model(features)

        return self.classifier(temporal_features)
        
