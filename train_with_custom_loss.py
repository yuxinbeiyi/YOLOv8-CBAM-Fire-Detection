import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from ultralytics import YOLO
from custom_losses import EIouLoss, FocalLoss

# 注册 CBAM 自定义模块到 ultralytics 命名空间
from cbam import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.tasks as _nn_tasks
_nn_tasks.CBAM = CBAM
_nn_tasks.ChannelAttention = ChannelAttention
_nn_tasks.SpatialAttention = SpatialAttention

class CustomLossWrapper:
    """自定义损失函数包装器"""
    
    def __init__(self):
        self.eiou_loss = EIouLoss()
        self.focal_loss = FocalLoss()
        
    def compute_bbox_loss(self, pred_boxes, target_boxes):
        """计算边界框损失"""
        return self.eiou_loss(pred_boxes, target_boxes)
    
    def compute_cls_loss(self, pred_scores, target_scores):
        """计算分类损失"""
        return self.focal_loss(pred_scores, target_scores)

class CustomYOLO:
    """自定义YOLO模型类"""
    
    def __init__(self, model_cfg='yolov8m-cbam-neck.yaml', weights=None):
        self.model_cfg = model_cfg
        self.weights = weights
        self.model = None
        
    def build_model(self):
        """构建自定义模型"""
        # 加载基础模型
        if self.weights:
            self.model = YOLO(self.weights)
        else:
            self.model = YOLO(self.model_cfg)
        
        # 创建自定义损失函数实例
        self.custom_loss = CustomLossWrapper()
        
        return self.model
    
    def train(self, **kwargs):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        # 设置训练参数
        train_kwargs = {
            'data': 'data.yaml',
            'imgsz': 640,
            'epochs': 200,
            'batch': 16,
            'single_cls': False,
            'workers': 8,
            'device': '0',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.5,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.15,
            'dropout': 0.0,
            'label_smoothing': 0.0,
            'patience': 50,
            'pretrained': True,
            'optimizer': 'SGD',
            'cos_lr': True,
            'close_mosaic': 15,
        }
        
        # 更新用户自定义参数
        train_kwargs.update(kwargs)
        
        # 开始训练
        results = self.model.train(**train_kwargs)
        return results

if __name__ == '__main__':
    # 创建自定义YOLO实例
    custom_yolo = CustomYOLO(model_cfg='yolov8m-cbam-neck.yaml')
    
    # 开始训练
    results = custom_yolo.train()
    
    print("训练完成！")
    print(f"最佳mAP: {results.box.map}")
    print(f"最佳mAP50: {results.box.map50}")
    print(f"最佳mAP75: {results.box.map75}")