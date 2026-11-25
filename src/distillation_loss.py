#!/usr/bin/env python3
"""
================================================================================
Unified Distillation Loss Module (PyTorch 2.x)
================================================================================
Implements:
1. Label-Smoothed Task Cross-Entropy & Metric Margin Loss (Deng et al. ECCV 2020)
2. Response-Based Knowledge Distillation with T^2 Scaling (Hinton et al. 2015)
3. Contrastive Representation Distillation InfoNCE (Tian et al. ICLR 2020)
4. Masked Relational Knowledge Distillation: Distance & Angle (Park et al. CVPR 2019)
================================================================================
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedDistillationLoss(nn.Module):
    """
    Composite Multi-Objective Distillation Loss for HCCR:
    L_total = L_task + α*L_KD(T) + β*L_CRD(τ) + γ*L_R-dist + δ*L_R-angle
    """
    def __init__(
        self,
        T: float = 3.5,
        alpha: float = 0.70,
        beta_crd: float = 0.10,
        gamma_dist: float = 1.00,
        delta_angle: float = 2.00,
        tau: float = 0.07,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta_crd = beta_crd
        self.gamma_dist = gamma_dist
        self.delta_angle = delta_angle
        self.tau = tau
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def rkd_dist(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        """
        Distance-wise Relational Knowledge Distillation (Park et al. CVPR 2019).
        Normalizes Euclidean distance matrices by their mean non-diagonal distance.
        Explicitly masks out diagonal entries (i == j).
        """
        B = fs.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=fs.device)

        # Pairwise Euclidean distances [B, B]
        ds = torch.norm(fs.unsqueeze(1) - fs.unsqueeze(0), p=2, dim=2)
        dt = torch.norm(ft.unsqueeze(1) - ft.unsqueeze(0), p=2, dim=2)

        # Boolean mask excluding diagonal self-distances
        mask = ~torch.eye(B, dtype=torch.bool, device=fs.device)
        ds_off = ds[mask]
        dt_off = dt[mask]

        mu_s = ds_off.mean() + 1e-7
        mu_t = dt_off.mean() + 1e-7

        return F.smooth_l1_loss(ds_off / mu_s, dt_off / mu_t)

    def rkd_angle(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        """
        Angle-wise Relational Knowledge Distillation (Park et al. CVPR 2019).
        Transfers the angular curvature of embedding triplets (e_ij, e_kj).
        Explicitly masks out degenerate triplets (i == j, k == j, i == k).
        """
        B = fs.size(0)
        if B <= 2:
            return torch.tensor(0.0, device=fs.device)

        # Difference vectors [j, i, D]
        diff_s = fs.unsqueeze(0) - fs.unsqueeze(1)
        diff_t = ft.unsqueeze(0) - ft.unsqueeze(1)

        norm_s = torch.norm(diff_s, p=2, dim=2, keepdim=True) + 1e-12
        norm_t = torch.norm(diff_t, p=2, dim=2, keepdim=True) + 1e-12
        td_s = diff_s / norm_s
        td_t = diff_t / norm_t

        # Batch matrix multiplication over feature dimension [B, B, B]
        angle_s = torch.bmm(td_s, td_s.transpose(1, 2))
        angle_t = torch.bmm(td_t, td_t.transpose(1, 2))

        # Strictly mask out degenerate triplets
        mask = torch.ones(B, B, B, dtype=torch.bool, device=fs.device)
        eye = torch.eye(B, dtype=torch.bool, device=fs.device)
        mask[eye.unsqueeze(2).expand(B, B, B)] = False
        mask[eye.unsqueeze(1).expand(B, B, B)] = False
        mask[eye.unsqueeze(0).expand(B, B, B)] = False

        return F.smooth_l1_loss(angle_s[mask], angle_t[mask])

    def crd_loss(self, fs_norm: torch.Tensor, ft_norm: torch.Tensor) -> torch.Tensor:
        """
        Contrastive Representation Distillation via Mini-Batch InfoNCE (Tian et al. ICLR 2020).
        Maximizes Mutual Information I(f_S; f_T) between student and teacher representations.
        Pulls positive pair (s_i, t_i) while repelling negative pairs (s_i, t_j) for j != i.
        """
        B = fs_norm.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=fs_norm.device)
            
        sim_matrix = torch.matmul(fs_norm, ft_norm.T) / self.tau  # [B, B]
        targets = torch.arange(B, device=fs_norm.device)
        return F.cross_entropy(sim_matrix, targets)

    def forward(
        self,
        s_logits: torch.Tensor,
        s_feats: torch.Tensor,
        t_logits: torch.Tensor,
        t_feats: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the composite distillation loss.
        Args:
            s_logits: Student combined logits [B, 15]
            s_feats:  Student 256-d latent features [B, 256]
            t_logits: Teacher ensemble logits [B, 15]
            t_feats:  Teacher ensemble 256-d latent features [B, 256]
            targets:  Ground-truth integer class labels [B]
        """
        # 1. Supervised Task Cross-Entropy Loss
        l_task = self.ce(s_logits, targets)

        # 2. Temperature-Scaled Response Distillation Loss (Hinton et al. 2015)
        s_soft = F.log_softmax(s_logits / self.T, dim=1)
        t_soft = F.softmax(t_logits / self.T, dim=1)
        l_kd = self.kld(s_soft, t_soft) * (self.T * self.T)

        # 3. Latent Feature Normalization onto Unit Hypersphere S^{255}
        s_norm = F.normalize(s_feats, p=2, dim=1)
        t_norm = F.normalize(t_feats, p=2, dim=1)

        # 4. Contrastive Representation Distillation (InfoNCE)
        l_crd = self.crd_loss(s_norm, t_norm)

        # 5. Masked Relational Knowledge Distillation (Distance & Angle)
        l_dist = self.rkd_dist(s_norm, t_norm)
        l_angle = self.rkd_angle(s_norm, t_norm)

        # Composite Weighted Sum
        total = (
            l_task
            + self.alpha * l_kd
            + self.beta_crd * l_crd
            + self.gamma_dist * l_dist
            + self.delta_angle * l_angle
        )

        metrics = {
            'loss_total': total.item(),
            'loss_task': l_task.item(),
            'loss_kd': l_kd.item(),
            'loss_crd': l_crd.item(),
            'loss_rkd_dist': l_dist.item(),
            'loss_rkd_angle': l_angle.item()
        }

        return total, metrics
