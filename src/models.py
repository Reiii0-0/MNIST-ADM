import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixStyle(nn.Module):
    """
    MixStyle: Domain Generalization with MixStyle (Zhou et al., ICLR 2021).
    Mixes feature statistics (mean and std) across different writers during training.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B)

        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)

        return x_normed * sig_mix + mu_mix

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAMBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_CBAM(nn.Module):
    """
    ResNet-18 tailored for 64x64 1-channel Grayscale Chinese Characters with CBAM & MixStyle.
    """
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True):
        super().__init__()
        self.in_planes = 64
        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(p=0.5, alpha=0.1) if use_mixstyle else nn.Identity()

        # Initial conv for 64x64 (using 3x3 kernel instead of 7x7 to preserve fine stroke details)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockCBAM(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlockCBAM.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_mixstyle:
            out = self.mixstyle(out)
        out = self.layer1(out)
        if self.use_mixstyle:
            out = self.mixstyle(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        feats = F.relu(self.fc(out))
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats

class SubCenterArcFace(nn.Module):
    """
    Sub-center ArcFace (Deng et al., ECCV 2020)
    Allows K sub-centers per class to resolve multi-modal distributions (e.g. Simplified vs Traditional Hanzi).
    """
    def __init__(self, in_features, num_classes=15, sub_centers=2, s=32.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        # features: (B, in_features)
        x = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=2)  # (C, K, in_features)

        # Cosine similarity: (B, C, K)
        cosine = torch.einsum('bd,ckd->bck', x, w)
        # Max pool over sub-centers: (B, C)
        cosine_max, _ = torch.max(cosine, dim=2)

        if labels is None or not self.training:
            return cosine_max * self.s

        # Apply angular margin to the target class
        sine = torch.sqrt(1.0 - torch.clamp(cosine_max ** 2, 0.0, 1.0))
        phi = cosine_max * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_max > self.th, phi, cosine_max - self.mm)

        one_hot = torch.zeros(cosine_max.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_max)
        output *= self.s
        return output

class CompactCNN(nn.Module):
    """
    Compact 4-Layer CNN baseline for small 64x64 images.
    """
    def __init__(self, in_channels=1, num_classes=15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits, feat

class LayerNorm2d(nn.Module):
    """LayerNorm that supports (N, C, H, W) format (channels first)."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (Liu et al., CVPR 2022).
    7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv (4x expand) -> GELU -> 1x1 Conv.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input + x

class ConvNeXt_Femto(nn.Module):
    """
    ConvNeXt-Femto architecture adapted for 64x64 Grayscale Chinese Characters.
    Stage dims: (48, 96, 192, 384), Depths: (2, 2, 4, 2).
    """
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, depths=(2, 2, 4, 2), dims=(48, 96, 192, 384)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem: 64x64 -> 32x32
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        # 3 Downsampling stages
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = LayerNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dims[-1], feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feats = F.gelu(self.fc(x))
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats

