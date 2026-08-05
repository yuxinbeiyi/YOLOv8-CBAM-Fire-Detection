import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss
from custom_losses import EIouLoss, FocalLoss, VarifocalLoss

class CustomBboxLoss(BboxLoss):
    """自定义边界框损失函数，支持EIoU"""
    
    def __init__(self, reg_max, use_dfl=False, iou_type='eiou'):
        super().__init__(reg_max, use_dfl)
        self.iou_type = iou_type
        self.eiou_loss = EIouLoss()
        
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """重写前向传播"""
        if self.iou_type == 'eiou':
            # 使用EIoU损失
            loss = self.eiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            return loss
        else:
            # 使用默认的IoU损失
            return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

def train_with_custom_loss(
    model_cfg='yolov8m-test.yaml',
    data_cfg='data.yaml',
    weights=None,
    epochs=100,
    batch_size=16,
    img_size=640,
    loss_config=None,
    **train_kwargs
):
    """
    使用自定义损失函数训练YOLOv8模型
    
    Args:
        model_cfg: 模型配置文件路径
        data_cfg: 数据集配置文件路径
        weights: 预训练权重路径
        epochs: 训练轮次
        batch_size: 批次大小
        img_size: 输入图像尺寸
        loss_config: 损失函数配置字典
        **train_kwargs: 其他训练参数
    """
    
    # 默认损失配置
    if loss_config is None:
        loss_config = {
            'bbox_loss': 'eiou',  # eiou, ciou, diou, giou
            'cls_loss': 'bce',    # bce, focal, varifocal
            'obj_loss': 'bce',    # bce, focal
        }
    
    # 加载模型
    if weights:
        model = YOLO(weights)
    else:
        model = YOLO(model_cfg)
    
    # 基础训练参数
    base_config = {
        'data': data_cfg,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'single_cls': True,
        'workers': 8,
        'device': '0',
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
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
        'dropout': 0.2,
        'pretrained': True if not weights else False,
        'optimizer': 'auto',
        'cos_lr': True,
        'close_mosaic': 10,
    }
    
    # 根据损失配置调整参数
    if loss_config.get('bbox_loss') == 'eiou':
        # EIoU需要较小的学习率
        base_config['lr0'] = 0.005
        base_config['lrf'] = 0.05
    
    if loss_config.get('cls_loss') in ['focal', 'varifocal']:
        # Focal Loss相关配置
        base_config['label_smoothing'] = 0.1
    
    # 更新用户自定义参数
    base_config.update(train_kwargs)
    
    # 开始训练
    print(f"开始训练，使用损失配置: {loss_config}")
    print(f"训练参数: {base_config}")
    
    results = model.train(**base_config)
    
    return results, model

def compare_loss_functions():
    """比较不同损失函数的效果"""
    
    loss_configs = [
        {
            'name': 'baseline',
            'bbox_loss': 'ciou',
            'cls_loss': 'bce',
            'obj_loss': 'bce'
        },
        {
            'name': 'eiou_only',
            'bbox_loss': 'eiou',
            'cls_loss': 'bce',
            'obj_loss': 'bce'
        },
        {
            'name': 'eiou_focal',
            'bbox_loss': 'eiou',
            'cls_loss': 'focal',
            'obj_loss': 'bce'
        },
        {
            'name': 'eiou_varifocal',
            'bbox_loss': 'eiou',
            'cls_loss': 'varifocal',
            'obj_loss': 'bce'
        }
    ]
    
    results = {}
    
    for config in loss_configs:
        print(f"\n=== 训练配置: {config['name']} ===")
        
        # 使用较少的epochs进行快速比较
        train_results, model = train_with_custom_loss(
            epochs=50,  # 快速比较
            batch_size=8,
            loss_config=config
        )
        
        results[config['name']] = {
            'map50': train_results.box.map50,
            'map': train_results.box.map,
            'model': model
        }
        
        print(f"{config['name']} - mAP50: {train_results.box.map50:.3f}, mAP: {train_results.box.map:.3f}")
    
    # 输出比较结果
    print("\n=== 损失函数比较结果 ===")
    for name, result in results.items():
        print(f"{name}: mAP50={result['map50']:.3f}, mAP={result['map']:.3f}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8自定义损失函数训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'compare'], 
                       help='训练模式: train-单次训练, compare-比较不同损失函数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--bbox-loss', type=str, default='eiou', 
                       choices=['ciou', 'diou', 'giou', 'eiou'], help='边界框损失函数')
    parser.add_argument('--cls-loss', type=str, default='bce', 
                       choices=['bce', 'focal', 'varifocal'], help='分类损失函数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 单次训练
        loss_config = {
            'bbox_loss': args.bbox_loss,
            'cls_loss': args.cls_loss,
            'obj_loss': 'bce'
        }
        
        results, model = train_with_custom_loss(
            epochs=args.epochs,
            batch_size=args.batch,
            loss_config=loss_config
        )
        
        print(f"\n训练完成!")
        print(f"最佳mAP50: {results.box.map50:.3f}")
        print(f"最佳mAP: {results.box.map:.3f}")
        
        # 保存模型
        model_path = f"yolov8m_custom_loss_{args.bbox_loss}_{args.cls_loss}.pt"
        model.export(format='pt')  # 保存为PyTorch格式
        print(f"模型已保存: {model_path}")
        
    elif args.mode == 'compare':
        # 比较不同损失函数
        compare_loss_functions()