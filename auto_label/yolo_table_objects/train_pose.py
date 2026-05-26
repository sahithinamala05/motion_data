"""Fine-tune YOLOv8n-pose on the table-objects keypoint dataset
(discard_tray / shoe / spare_shoe / scanner, 7 kpts each).
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data_pose.yaml'
RUNS = ROOT / 'runs'


def main():
    model = YOLO('yolov8n-pose.pt')
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=1280,
        batch=8,
        device=0,
        patience=25,
        project=str(RUNS),
        name='table_objects_pose',
        exist_ok=True,
        seed=42,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        scale=0.1,
        translate=0.05,
        mosaic=0.5,
        close_mosaic=10,
    )


if __name__ == '__main__':
    main()
