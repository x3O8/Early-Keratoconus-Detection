"""
Multi-modal image architecture (matches utils/model_loader.py — no Streamlit).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

MODALITIES = ["CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P"]
NUM_MODALITIES = len(MODALITIES)


class ModalityEncoder(nn.Module):
    """Single-modality ResNet encoder."""

    def __init__(self, embed_dim: int = 256, pretrained: bool = False):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        return self.proj(pooled), feat


class MultiModalKeratoconusModel(nn.Module):
    """Seven ophthalmic modalities → cross-modal fusion → classifier."""

    def __init__(self, num_modalities: int = 7, embed_dim: int = 256, num_heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim

        self.encoders = nn.ModuleList([ModalityEncoder(embed_dim) for _ in range(num_modalities)])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.modality_attn = nn.Linear(embed_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: list):
        embeds = []
        spatial_feats = []
        for i, img in enumerate(images):
            embed, spatial = self.encoders[i](img)
            embeds.append(embed.unsqueeze(1))
            spatial_feats.append(spatial)

        tokens = torch.cat(embeds, dim=1)
        tokens = self.transformer(tokens)
        attn_scores = self.modality_attn(tokens).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(context)
        return logits, attn_weights, spatial_feats
