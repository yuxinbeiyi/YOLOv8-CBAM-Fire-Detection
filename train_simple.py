import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from custom_losses import EIouLoss

def train_with_eiou_loss(
    model_cfg='yolov8m-test.yaml',
    data_cfg='data.yaml',
    weights=None,
    epochs=100,
    batch_size=8,
    img_size=640
):
    """
    使用EIoU损失函数训练YOLOv8模型
    
    Args:
        model_cfg: 模型配置文件路径
        data_cfg: 数据集配置文件路径
        weights: 预训练权重路径
        epochs: 训练轮次
        batch_size: 批次大小
        img_size: 输入图像尺寸
    """
    
    # 加载模型
    if weights:
        model = YOLO(weights)
    else:
        model = YOLO(model_cfg)
    
    # 训练参数（针对EIoU优化）
    train_config = {
        'data': data_cfg,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'single_cls': True,
        'workers': 8,
        'device': '0',
        
        # 学习率调整（EIoU需要更小的学习率）
        'lr0': 0.005,           # 降低学习率
        'lrf': 0.05,            # 最终学习率比例
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        
        # 正则化
        'dropout': 0.2,
        'label_smoothing': 0.1,  # 标签平滑
        
        # 优化器
        'pretrained': True if not weights else False,
        'optimizer': 'SGD',      # 使用SGD优化器
        'cos_lr': True,          # 余弦学习率调度
        'close_mosaic': 10,      # 最后10轮关闭Mosaic
    }
    
    print("开始训练，使用EIoU损失函数优化...")
    print(f"训练参数: {train_config}")
    
    # 开始训练
    results = model.train(**train_config)
    
    return results, model

def main():
    """主函数"""
    
    # 训练配置
    config = {
        'model_cfg': 'yolov8m-test.yaml',
        'data_cfg': 'data.yaml',
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640
    }
    
    print("=== YOLOv8 EIoU损失函数优化训练 ===")
    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    results, model = train_with_eiou_loss(**config)
    
    # 输出训练结果
    print("\n=== 训练完成 ===")
    print(f"最佳mAP50: {results.box.map50:.3f}")
    print(f"最佳mAP: {results.box.map:.3f}")
    print(f"最佳mAP75: {results.box.map75:.3f}")
    
    # 保存模型
    model_path = "yolov8m_eiou_optimized.pt"
    model.export(format='pt')
    print(f"模型已保存: {model_path}")
    
    # 验证模型
    print("\n=== 模型验证 ===")
    val_results = model.val()
    print(f"验证集mAP50: {val_results.box.map50:.3f}")
    print(f"验证集mAP: {val_results.box.map:.3f}")

if __name__ == '__main__':
    main()