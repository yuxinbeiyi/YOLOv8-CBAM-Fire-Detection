import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from custom_losses import EIouLoss, FocalLoss

# ============================================================
# CBAM 注意力机制注册
# ultralytics parse_model 通过 globals()[m] 查找模块，
# 因此必须在训练启动前将 CBAM 注入 ultralytics.nn.tasks 命名空间
# ============================================================
from cbam import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.tasks as _nn_tasks
_nn_tasks.CBAM = CBAM
_nn_tasks.ChannelAttention = ChannelAttention
_nn_tasks.SpatialAttention = SpatialAttention


def de_parallel(model):
    """兼容新版 ultralytics：从 DataParallel 包装中取出原始模型"""
    return model.module if hasattr(model, 'module') else model


# ============================================================
# 1. 自定义边界框损失：用 EIoU 替换内置 BboxLoss
# ============================================================
class CustomBboxLoss(BboxLoss):
    """将内置 BboxLoss 中的 IoU 计算替换为 EIoU"""

    def __init__(self, reg_max, use_dfl=False):
        super().__init__(reg_max)   # 8.3.18 BboxLoss 只接受 reg_max
        self.use_dfl = use_dfl      # 手动保存，供 forward 使用
        self.eiou_loss = EIouLoss()

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        # EIoU 边界框损失（仅对前景 anchor 计算）
        if fg_mask.sum() > 0:
            bbox_loss = self.eiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            bbox_loss = bbox_loss / target_scores_sum
        else:
            bbox_loss = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        # DFL 分布式焦点损失（保留原始实现）
        if self.use_dfl and fg_mask.sum() > 0:
            dist_loss = self._df_loss(
                pred_dist[fg_mask].view(-1, self.reg_max + 1),
                target_bboxes[fg_mask]
            ) / target_scores_sum
        else:
            dist_loss = torch.tensor(0.0, device=pred_bboxes.device)

        return bbox_loss, dist_loss


# ============================================================
# 2. 自定义检测损失：将 CustomBboxLoss 注入 v8DetectionLoss
# ============================================================
class CustomDetectionLoss(v8DetectionLoss):
    """继承 v8DetectionLoss，替换其内部的 BboxLoss 为 EIoU 版本"""

    def __init__(self, model):
        super().__init__(model)
        # 用 EIoU 版本替换内置的 BboxLoss
        self.bboxloss = CustomBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl)


# ============================================================
# 3. 自定义 Trainer：在训练初始化完成后注入自定义损失
# ============================================================
class CustomDetectionTrainer(DetectionTrainer):
    """子类化 DetectionTrainer，将 compute_loss 替换为自定义 EIoU 损失"""

    def _setup_train(self, world_size):
        # 先执行父类完整的训练初始化（包括默认 compute_loss 的设置）
        super()._setup_train(world_size)
        # 在父类初始化之后，替换 compute_loss 为自定义版本
        self.compute_loss = CustomDetectionLoss(de_parallel(self.model))
        print('✅ 自定义 EIoU 损失函数已成功注入训练管道')


# ============================================================
# 4. 训练入口
# ============================================================
if __name__ == '__main__':
    # ★ train34：以 train33/best.pt 为起点，温和 fine-tuning 压制过拟合 + 提升 Recall
    # train33 分析结论：mAP50=0.5806 创历史新高（epoch 84），Precision=0.6746 历史最高。
    # 但 epoch 84 后出现与 train31 相同的过拟合规律（val_box_loss +4.9%），
    # Recall 仍偏低（0.518），Precision-Recall 差距达 15.6 个百分点。
    # 本轮核心策略：温和 fine-tuning，改动幅度严格控制，避免重蹈 train32 覆辙。
    # 核心改动：
    #   1. lr0 0.001 → 0.0003：大幅降低，精细打磨 train33 最优权重
    #   2. lrf 0.01 → 0.05：提高衰减终值比例，保留更多探索空间
    #   3. cls 0.5 → 0.4：温和下调（非 train32 的激进 0.3），引导提升 Recall
    #   4. dropout 0.1 → 0.15：小幅增强正则化，压制过拟合
    #   5. weight_decay 0.0005 → 0.001：适度加强 L2 正则化
    #   6. patience 80 → 50：节省算力，避免过拟合后空跑
    #   7. warmup_epochs 3.0 → 1.0：fine-tuning 阶段热身期缩短
    #   8. box/dfl 保持不变（7.5/1.5）：不再动损失权重平衡
    # 上传 AutoDL 时请将 train33/weights/best.pt 重命名为 train33_best.pt
    # 放在与 train.py 相同目录下
    trainer = CustomDetectionTrainer(overrides={
        # ===== 模型：fine-tuning 起点 =====
        'model': 'train33_best.pt',    # 从 train33 历史最优权重继续（epoch 84）
        'pretrained': False,            # 已有完整权重，无需重新加载预训练
        'data': 'data.yaml',
        'imgsz': 640,
        'epochs': 200,
        'patience': 50,                # 节省算力
        'batch': 16,
        'single_cls': False,
        'workers': 8,
        'device': '0',

        # ===== 学习率（fine-tuning 精细打磨）=====
        'lr0': 0.0003,             # ★ 0.001 → 0.0003：精细微调起始 lr
        'lrf': 0.05,               # 终止 lr = 0.0003×0.05 = 1.5e-5
        'weight_decay': 0.001,     # ★ 0.0005 → 0.001：适度加强 L2 正则化
        'warmup_epochs': 1.0,      # fine-tuning 热身期缩短

        # ===== NMS 阈值 =====
        'iou': 0.6,

        # ===== 数据增强 =====
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,

        # ===== 损失权重（温和调整，仅动 cls）=====
        'cls': 0.4,                # ★ 0.5 → 0.4：温和下调，引导提升 Recall
        'box': 7.5,                # 保持不变，避免重蹈 train32 覆辙
        'dfl': 1.5,                # 保持不变

        # ===== 正则化与优化 =====
        'dropout': 0.15,           # ★ 0.1 → 0.15：小幅增强，压制过拟合
        'label_smoothing': 0.1,
        'optimizer': 'AdamW',
        'cos_lr': True,
        'close_mosaic': 20,        # 适度延长稳定期
    })

    trainer.train()
