import warnings

warnings.filterwarnings("ignore")
import torch
import ultralytics.nn.tasks as _nn_tasks
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss

# ============================================================
# CBAM 注意力机制注册（必须在任何模型加载之前执行）
# ============================================================
from cbam import CBAM, ChannelAttention, SpatialAttention
from custom_losses import EIouLoss

_nn_tasks.CBAM = CBAM
_nn_tasks.ChannelAttention = ChannelAttention
_nn_tasks.SpatialAttention = SpatialAttention


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


# ============================================================
# 1. 自定义边界框损失：EIoU 替换内置 BboxLoss
# ============================================================
class CustomBboxLoss(BboxLoss):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__(reg_max)
        self.use_dfl = use_dfl
        self.eiou_loss = EIouLoss()

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        if fg_mask.sum() > 0:
            bbox_loss = self.eiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            bbox_loss = bbox_loss / target_scores_sum
        else:
            bbox_loss = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        if self.use_dfl and fg_mask.sum() > 0:
            dist_loss = (
                self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_bboxes[fg_mask]) / target_scores_sum
            )
        else:
            dist_loss = torch.tensor(0.0, device=pred_bboxes.device)

        return bbox_loss, dist_loss


# ============================================================
# 2. 自定义检测损失
# ============================================================
class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bboxloss = CustomBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl)


# ============================================================
# 3. 自定义 Trainer：
#    - _setup_train 中从 yolov8m.pt 部分加载 backbone 权重
#    - 注入 EIoU 损失
# ============================================================
class CustomDetectionTrainer(DetectionTrainer):
    def _setup_train(self, world_size):
        # 先执行父类完整初始化（模型从 yolov8m-cbam-neck.yaml 构建）
        super()._setup_train(world_size)

        # ----------------------------------------------------------
        # 从 yolov8m.pt 加载与 CBAM 模型层名和尺寸均匹配的权重
        # 不匹配的层（即 CBAM 模块）保持随机初始化
        # 这样 backbone 权重与 baseline 完全一致，确保对比公平
        # ----------------------------------------------------------
        ckpt = torch.load("yolov8m.pt", map_location="cpu")
        pretrained_sd = ckpt["model"].float().state_dict()
        model_sd = de_parallel(self.model).state_dict()

        matched, total = 0, len(model_sd)
        update_sd = {}
        for k, v in pretrained_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                update_sd[k] = v
                matched += 1

        model_sd.update(update_sd)
        de_parallel(self.model).load_state_dict(model_sd, strict=False)
        print(f"[权重加载] 从 yolov8m.pt 匹配并加载 {matched}/{total} 层（未匹配的 CBAM 模块保持随机初始化）")

        # 注入 EIoU 损失
        self.compute_loss = CustomDetectionLoss(de_parallel(self.model))
        print("[损失函数] EIoU 损失已注入训练管道")


# ============================================================
# 4. 训练入口
# 所有训练参数与 train_baseline 完全一致，只改变模型和损失函数
# ============================================================
if __name__ == "__main__":
    trainer = CustomDetectionTrainer(
        overrides={
            # ===== 模型：CBAM 架构，从 yaml 构建，backbone 权重在 _setup_train 中手动加载 =====
            "model": "yolov8m-cbam-neck.yaml",
            "pretrained": False,  # 权重由 _setup_train 手动处理，此处关闭自动加载
            "data": "data.yaml",
            "imgsz": 640,
            "epochs": 200,
            "patience": 50,
            "batch": 16,
            "single_cls": False,
            "workers": 8,
            "device": "0",
            "seed": 0,
            "deterministic": True,
            # ===== 优化器（与 baseline 完全一致）=====
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            # ===== 损失权重（与 baseline 完全一致）=====
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "label_smoothing": 0.0,
            "iou": 0.7,
            # ===== 数据增强（与 baseline 完全一致）=====
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.5,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.15,
            "auto_augment": "randaugment",
            "erasing": 0.4,
            "dropout": 0.0,
            "close_mosaic": 15,
            # ===== 其他（与 baseline 完全一致）=====
            "amp": True,
            "name": "train_improved",
        }
    )

    trainer.train()
