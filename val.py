
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('runs/detect/train5/weights/best.pt')
    model.val(data='data.yaml',
                imgsz=640,
                batch=16,
                split='val',            # 注释掉data.yaml的test的话，这里应该将test改为val
                workers=10,
                device='0',
                )
                    
