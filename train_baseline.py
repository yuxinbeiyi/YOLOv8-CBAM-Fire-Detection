"""
纯YOLOv8m基线训练脚本（无CBAM、无自定义损失）
用于消融实验对照组，超参数与train34保持一致.
"""

import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载标准YOLOv8m预训练权重（无任何结构改动）
    model = YOLO("yolov8m.pt")

    # 训练参数与train34完全对齐，确保消融实验公平性
    results = model.train(
        data="data.yaml",
        imgsz=640,
        epochs=200,
        batch=16,
        single_cls=False,
        workers=8,
        device="0",
        # 优化器（与train34一致）
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        # 数据增强（与train34一致）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        close_mosaic=15,
        # 其他（与train34一致）
        dropout=0.0,
        label_smoothing=0.0,
        patience=50,
        pretrained=True,
        # 结果保存到单独目录，便于区分
        name="train_baseline",
    )

    print("基线训练完成！")
    print(f"最佳mAP@50-95: {results.box.map:.4f}")
    print(f"最佳mAP@50:    {results.box.map50:.4f}")
    print(f"最佳mAP@75:    {results.box.map75:.4f}")
