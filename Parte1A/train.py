"""Fine-tuning YOLOv11n-seg sobre dataset propio (traffic_cone + traffic_highway + us_road_signs,
fusionado y convertido a segmentación en merge_dataset.py). Práctica 4-1, Parte 1A."""
import time
import glob
import shutil
import os

import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = "dataset_traffic_objects"
yaml_path = os.path.join(HERE, DATASET, "data.yaml")

model = YOLO(os.path.join(HERE, "models", "yolo11n-seg.pt"))
print(f"YOLOv11n-seg cargado | Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"Parámetros: {sum(p.numel() for p in model.model.parameters())/1e6:.1f}M")

t0 = time.time()
results = model.train(
    data=yaml_path,
    epochs=40,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu",
    project=os.path.join(HERE, "runs"),
    name="traffic_objects_v1",
    exist_ok=True,
    patience=10,
    save=True,
    plots=True,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    augment=True,
    degrees=10.0,
    fliplr=0.5,
    mosaic=1.0,
    verbose=False,
)

candidates = glob.glob(os.path.join(HERE, "runs", "**", "best.pt"), recursive=True)
if candidates:
    shutil.copy2(sorted(candidates)[-1], os.path.join(HERE, "models", "best.pt"))

print(f"\nEntrenamiento completado en {(time.time()-t0)/60:.1f} minutos")
print("Modelo disponible en: models/best.pt")
