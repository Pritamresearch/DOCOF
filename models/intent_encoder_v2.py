import torch
import torch.nn as nn


class IntentEncoderV2(nn.Module):
    """
    Encodes object tokens and predicts intent.
    """

    def __init__(self, d_model=128, num_classes=5):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                batch_first=True
            ),
            num_layers=2
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, object_tokens):
        """
        object_tokens: [B, N, D]
        """
        features = self.transformer(object_tokens)   # [B, N, D]
        logits = self.classifier(features)           # [B, N, C]

        return logits, features