import math

import torch
import torch.nn.functional as F
from torch import nn


class EIouLoss(nn.Module):
    """Enhanced IoU Loss for better bounding box regression."""

    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4] (x1, y1, x2, y2)
            target: [N, 4] (x1, y1, x2, y2).
        """
        # 确保坐标有效性
        pred = pred.clamp(min=0)
        target = target.clamp(min=0)

        # 计算交集
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # 计算并集
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area + self.eps

        # 计算IoU
        iou = inter_area / union_area

        # 计算中心点距离惩罚
        pred_center_x = (pred[:, 0] + pred[:, 2]) / 2
        pred_center_y = (pred[:, 1] + pred[:, 3]) / 2
        target_center_x = (target[:, 0] + target[:, 2]) / 2
        target_center_y = (target[:, 1] + target[:, 3]) / 2

        center_dist = (pred_center_x - target_center_x).pow(2) + (pred_center_y - target_center_y).pow(2)

        # 计算外接矩形
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])

        enclose_diag = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2) + self.eps

        # 计算宽高比惩罚
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

        aspect_ratio = (
            4
            / (math.pi**2)
            * (torch.atan(target_w / (target_h + self.eps)) - torch.atan(pred_w / (pred_h + self.eps))).pow(2)
        )

        # 计算EIoU
        eiou = iou - (center_dist / enclose_diag) - aspect_ratio
        loss = 1 - eiou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N, C] one-hot encoded or [N] class indices.
        """
        if targets.dim() == 1:
            # 如果是类别索引，转换为one-hot
            targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class VarifocalLoss(nn.Module):
    """Varifocal Loss for object detection."""

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_score, target_score):
        """
        Args:
            pred_score: predicted IOU-aware classification score
            target_score: target score (0 for negative, IoU for positive).
        """
        # Varifocal Loss公式
        pred_score = pred_score.sigmoid()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target_score).pow(self.gamma)

        loss = weight * F.binary_cross_entropy(pred_score, target_score, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss for joint classification and localization quality."""

    def __init__(self, beta=2.0, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, quality_label):
        """
        Args:
            pred: predicted classification scores [N, C]
            target: target labels [N] or [N, C]
            quality_label: quality score (e.g., IoU) [N].
        """
        # Quality Focal Loss实现
        pred_sigmoid = pred.sigmoid()
        scale_factor = pred_sigmoid - 0.5

        # 处理目标格式
        if target.dim() == 1:
            # 如果是类别索引，转换为one-hot
            target_one_hot = F.one_hot(target, num_classes=pred.size(-1)).float()
        else:
            target_one_hot = target

        # 扩展quality_label以匹配类别维度
        quality_label = quality_label.unsqueeze(-1).expand_as(pred)

        # 正样本损失
        pos_loss = (
            quality_label
            * scale_factor.abs().pow(self.beta)
            * F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred) * quality_label, reduction="none")
        )

        # 负样本损失
        neg_loss = (
            (1 - quality_label)
            * scale_factor.abs().pow(self.beta)
            * F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred), reduction="none")
        )

        # 根据目标标签选择损失
        loss = target_one_hot * pos_loss + (1 - target_one_hot) * neg_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
