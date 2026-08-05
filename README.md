# YOLOv8-CBAM-Fire-Detection

<div align="center">

**基于改进 YOLOv8 的火灾烟雾实时检测系统**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.18-00B4D8.svg)](https://github.com/ultralytics/ultralytics)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## 📖 项目简介

本项目针对火灾烟雾检测场景，基于 **YOLOv8m** 目标检测框架，引入 **CBAM（Convolutional Block Attention Module）注意力机制** 和 **EIoU（Efficient IoU）损失函数** 进行改进，构建了一个集**实时检测、可视化界面、邮件告警**于一体的火灾烟雾监测系统。

- **检测类别**：火焰（fire）、烟雾（smoke）
- **模型架构**：YOLOv8m + 7 个 CBAM 模块 + EIoU 损失
- **应用界面**：PyQt5 桌面 GUI，支持摄像头实时检测和本地文件检测

## 🏗️ 模型架构

### 整体设计

以 YOLOv8m 为基线模型，在 **Backbone** 和 **Neck** 两个阶段共嵌入 **7 个 CBAM 注意力模块**：

| 阶段       | 位置                | 通道数 | 说明                           |
| ---------- | ------------------- | ------ | ------------------------------ |
| Backbone   | P3 层后             | 192    | 小尺度特征注意力校准           |
| Backbone   | P4 层后             | 384    | 中尺度特征注意力校准           |
| Backbone   | SPPF 层后           | 576    | 大尺度特征注意力校准           |
| Neck (FPN) | Top-down P4 融合后  | 384    | 上采样融合特征过滤             |
| Neck (FPN) | Top-down P3 融合后  | 192    | 小尺度融合特征过滤 → Detect P3 |
| Neck (PAN) | Bottom-up P4 融合后 | 384    | 下采样融合特征过滤 → Detect P4 |
| Neck (PAN) | Bottom-up P5 融合后 | 576    | 大尺度融合特征过滤 → Detect P5 |

### 改进点

1. **CBAM 注意力机制**：在特征提取和特征融合的关键节点嵌入通道+空间双重注意力，使模型聚焦火焰/烟雾区域，抑制背景干扰
2. **EIoU 损失函数**：在 IoU 基础上引入中心点距离和宽高比惩罚，提升边界框回归精度

## 📊 消融实验结果

| 模型                            | mAP@50     | mAP@50-95  | Precision | Recall    |
| ------------------------------- | ---------- | ---------- | --------- | --------- |
| YOLOv8m（基线）                 | 0.5978     | 0.3495     | 0.666     | 0.537     |
| **YOLOv8m+CBAM+EIoU（本研究）** | **0.5998** | **0.3516** | **0.668** | **0.538** |

改进模型在全部四项指标上均优于基线模型，验证了 CBAM 注意力机制和 EIoU 损失函数的有效性。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+（GPU 训练推荐）

### 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.3.18
pip install opencv-python PyQt5
```

### 模型权重

训练好的模型权重文件需放置于项目根目录，命名为 `best.pt`。由于权重文件较大，不包含在代码仓库中。

### 运行检测系统

```bash
python main.py
```

启动后将打开 PyQt5 图形界面，支持：

- **摄像头实时检测**：自动枚举可用摄像头，支持热切换
- **本地图片/视频检测**：选择文件进行推理并输出结果
- **置信度阈值调节**：0.0 ~ 0.99 范围内动态调整
- **邮件告警**：检测到高置信度目标时，自动发送告警邮件（需配置 163 邮箱）

### 训练模型

```bash
# 公平对比实验（基线）
python train_baseline.py

# 改进模型训练（CBAM + EIoU）
python train_fair.py

# 微调训练
python train.py
```

## 📁 项目结构

```
yolov8/
├── main.py                       # PyQt5 GUI 检测主程序
├── train.py                      # 自定义训练脚本（EIoU + CBAM）
├── train_baseline.py             # 基线模型训练脚本
├── train_fair.py                 # 公平对比实验训练脚本
├── train_advanced.py             # 高级训练脚本（参数化损失配置）
├── train_simple.py               # 简化训练脚本
├── train_with_custom_loss.py     # 自定义损失训练脚本
├── detect.py                     # 检测脚本
├── val.py                        # 验证脚本
├── export_onnx.py                # ONNX 导出脚本
├── run.py                        # 快速启动脚本
├── cbam.py                       # CBAM 注意力机制实现
├── custom_losses.py              # 自定义损失函数（EIoU/Focal/Varifocal）
├── message.py                    # 邮件告警模块
├── yolov8m-cbam-neck.yaml        # 模型架构配置（7 个 CBAM）
├── yolov8m-cbam.yaml             # 模型架构配置（仅 Backbone CBAM）
├── yolov8m-test.yaml             # 基线模型配置
├── data.yaml                     # 数据集配置
└── best.pt                       # 训练好的模型权重（需自行放置）
```

## 🔧 核心模块说明

### CBAM 注意力机制 ([cbam.py](cbam.py))

实现 CBAM: Convolutional Block Attention Module (ECCV 2018)：

- **ChannelAttention**：avg+max 双路池化 → 共享 MLP → Sigmoid，学习通道重要性
- **SpatialAttention**：通道维度均值/最大值拼接 → 7×7 卷积 → Sigmoid，学习空间位置权重
- **CBAM**：先通道注意力，再空间注意力，均为残差乘法形式，不改变特征图维度

### 自定义损失函数 ([custom_losses.py](custom_losses.py))

- **EIouLoss**：IoU + 中心点距离惩罚 + 宽高比惩罚（实际用于训练）
- **FocalLoss**：解决类别不平衡问题
- **VarifocalLoss**：非对称焦点损失，兼顾正负样本
- **QualityFocalLoss**：结合 IoU 质量得分的焦点损失

### 邮件告警 ([message.py](message.py))

- 基于 163 邮箱 SMTP 服务
- 检测到高置信目标时自动发送告警邮件（含截图附件）
- 支持 20 秒冷却时间，防止重复告警

## 📝 配置说明

### 邮件配置

首次使用需在 GUI 界面中点击「邮件设置」配置 163 邮箱信息：

- **发件人**：163 邮箱地址
- **收件人**：接收告警的邮箱地址
- **授权码**：163 邮箱客户端授权码（非登录密码，需在 163 邮箱设置中开启 SMTP 服务获取）

配置将保存在 `email_config.json` 中，**该文件已加入 .gitignore，不会被提交到版本控制**。

### 数据集配置

数据集路径在 `data.yaml` 中配置，格式为 YOLOv8 标准格式：

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 2
names: ["fire", "smoke"]
```

## ⚠️ 注意事项

1. 首次运行需确保 `best.pt` 模型权重文件存在于项目根目录
2. 使用邮件告警功能需在 163 邮箱中开启 SMTP 服务并获取授权码
3. `email_config.json` 包含敏感信息，请勿提交到公共仓库
4. 训练数据集未包含在仓库中，需自行准备

## 📄 License

本项目基于 [Apache 2.0](LICENSE) 协议开源。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 目标检测框架
- [CBAM (ECCV 2018)](https://arxiv.org/abs/1807.06521) — 注意力机制论文
- [EIoU Loss](https://arxiv.org/abs/2101.08158) — 边界框回归损失
