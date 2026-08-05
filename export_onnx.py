"""
将训练好的 .pt 模型导出为 .onnx 格式
用途：供 CPU 环境下的用户使用，ONNX Runtime 推理速度显著快于 PyTorch CPU

使用方法：
    python export_onnx.py

导出成功后会在 PT_PATH 同目录下生成同名的 .onnx 文件
"""

import os

# ============================================================
# 1. 注册自定义 CBAM 模块（必须在加载模型前完成）
# ============================================================
from cbam import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.tasks as _nn_tasks
_nn_tasks.CBAM = CBAM
_nn_tasks.ChannelAttention = ChannelAttention
_nn_tasks.SpatialAttention = SpatialAttention

from ultralytics import YOLO

# ============================================================
# 2. 配置：修改此处路径指向你的 .pt 模型
# ============================================================
PT_PATH = r'E:/Desktop/yolo8远程/train33/weights/best.pt'

# ============================================================
# 3. 导出
# ============================================================
if not os.path.exists(PT_PATH):
    print(f'❌ 模型文件不存在，请检查路径：{PT_PATH}')
else:
    print(f'正在加载模型：{PT_PATH}')
    model = YOLO(PT_PATH)

    print('正在导出为 ONNX 格式，请稍候...')
    export_path = model.export(
        format='onnx',
        imgsz=640,       # 与训练时保持一致
        simplify=True,   # 简化计算图，提升推理速度
        opset=12,        # ONNX opset 版本，12 兼容性最好
    )

    print(f'✅ 导出成功：{export_path}')
    print('提示：将生成的 .onnx 文件路径填入 main.py 的 model_path 即可在 CPU 上运行')
