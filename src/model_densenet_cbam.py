#!/usr/bin/env python3
"""
DenseNet-121 CBAM with MixStyle and Sub-Center ArcFace for 64x64 HCCR.
Engineered as the 3rd complementary pillar for the Tri-Ensemble (ResNet18 + ConvNeXt-Femto + DenseNet121).

Architectural Highlights:
1. Specialized 64x64 Stem: 3x3 Conv with stride 1, preserving fragile stroke junctions (avoids standard 7x7 stride 2 downsampling).
2. Dense Feature Concatenation: Retains multi-scale receptive fields (RF = 3 to 63) directly to the classification head.
3. Dual-Attention CBAM (Channel + Spatial): Recalibrates multi-scale stroke features.
4. MixStyle (Zhou et al., ICLR 2021): Writer-invariant domain generalization.
5. Sub-Center ArcFace (K=2): Multi-modal prototype learning for cursive vs regular script.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import CBAMBlock, MixStyle, SubCenterArcFace


class DenseLayer(nn.Module):
    """
    Bottleneck Dense Layer: BN -> ReLU -> 1x1 Conv (4*k) -> BN -> ReLU -> 3x3 Conv (k)
    """
    def __init__(self, in_features, growth_rate, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate
        inter_channels = 4 * growth_rate

        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    """
    Dense Block containing multiple DenseLayers with dense feature concatenation.
    """
    def __init__(self, num_layers, in_features, growth_rate, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_features + i * growth_rate, growth_rate, drop_rate))
        self.block = nn.Sequential(*layers)
        self.out_features = in_features + num_layers * growth_rate

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    """
    Transition Layer: BN -> 1x1 Conv (compression theta=0.5) -> 2x2 AvgPool
    """
    def __init__(self, in_features, out_features, use_cbam=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cbam = CBAMBlock(out_features) if use_cbam else nn.Identity()

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x), inplace=True))
        out = self.cbam(out)
        out = self.pool(out)
        return out


class DenseNet121_CBAM(nn.Module):
    """
    DenseNet-121 tailored for 64x64 1-channel Grayscale Chinese Characters.
    Stage structure: (6, 12, 24, 16) layers with growth rate k=32.
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=15,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        compression=0.5,
        feature_dim=256,
        drop_rate=0.1,
        use_mixstyle=True
    ):
        super().__init__()
        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(p=0.5, alpha=0.1) if use_mixstyle else nn.Identity()

        # Custom 64x64 Stem (NO 7x7 conv and NO stride-2 maxpool)
        num_init_features = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )

        num_features = num_init_features
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            # Dense Block
            block = DenseBlock(num_layers, num_features, growth_rate, drop_rate=drop_rate)
            self.blocks.append(block)
            num_features = block.out_features

            # Transition layer (except after the last block)
            if i != len(block_config) - 1:
                out_trans = int(num_features * compression)
                trans = Transition(num_features, out_trans, use_cbam=True)
                self.transitions.append(trans)
                num_features = out_trans

        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_cbam = CBAMBlock(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(num_features, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        # Stem: 64x64 -> 64x64 (64 ch)
        x = self.stem(x)
        if self.use_mixstyle:
            x = self.mixstyle(x)

        # Dense Block 1: 64x64 -> 64x64 (256 ch)
        x = self.blocks[0](x)
        if self.use_mixstyle:
            x = self.mixstyle(x)
        # Transition 1: 64x64 -> 32x32 (128 ch)
        x = self.transitions[0](x)

        # Dense Block 2: 32x32 -> 32x32 (512 ch)
        x = self.blocks[1](x)
        # Transition 2: 32x32 -> 16x16 (256 ch)
        x = self.transitions[1](x)

        # Dense Block 3: 16x16 -> 16x16 (1024 ch)
        x = self.blocks[2](x)
        # Transition 3: 16x16 -> 8x8 (512 ch)
        x = self.transitions[2](x)

        # Dense Block 4: 8x8 -> 8x8 (1024 ch)
        x = self.blocks[3](x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = self.final_cbam(x)

        # GAP -> Feature Projection
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feats = F.relu(self.fc(x))
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats
