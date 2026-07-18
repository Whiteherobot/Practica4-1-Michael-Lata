"""Retoma el entrenamiento cortado desde runs/traffic_objects_v1/weights/last.pt."""
import os
import glob
import shutil
import time

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
LAST = os.path.join(HERE, "runs", "traffic_objects_v1", "weights", "last.pt")

model = YOLO(LAST)
t0 = time.time()
model.train(resume=True)

candidates = glob.glob(os.path.join(HERE, "runs", "**", "best.pt"), recursive=True)
if candidates:
    shutil.copy2(sorted(candidates)[-1], os.path.join(HERE, "models", "best.pt"))

print(f"\nEntrenamiento (retomado) completado en {(time.time()-t0)/60:.1f} minutos")
