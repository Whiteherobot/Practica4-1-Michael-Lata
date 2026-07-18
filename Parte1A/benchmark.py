"""Benchmark GPU vs CPU del modelo entrenado (Parte 1A). Corre despues de train.py."""
import glob
import os
import time

import numpy as np
import psutil
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best.pt")
DATASET = os.path.join(HERE, "dataset_traffic_objects")


def benchmark(model_path, images, device, n=50):
    m = YOLO(model_path)
    for img in images[:3]:
        m(img, device=device, verbose=False)
    times, ram = [], []
    for i in range(n):
        t0 = time.perf_counter()
        m(images[i % len(images)], device=device, verbose=False)
        times.append(time.perf_counter() - t0)
        ram.append(psutil.virtual_memory().used / 1024**3)
    return {"fps": 1 / np.mean(times), "ms": np.mean(times) * 1000, "ram": np.mean(ram)}


if __name__ == "__main__":
    test_imgs = glob.glob(os.path.join(DATASET, "test", "images", "*.jpg"))[:10]
    if not test_imgs:
        test_imgs = glob.glob(os.path.join(DATASET, "valid", "images", "*.jpg"))[:10]

    print(f"Validando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    val = model.val(data=os.path.join(DATASET, "data.yaml"), imgsz=640, verbose=False)
    print("=== METRICAS DE VALIDACION ===")
    print(f"mAP50 (Box):     {val.box.map50:.4f}")
    print(f"mAP50-95 (Box):  {val.box.map:.4f}")
    print(f"mAP50 (Mask):    {val.seg.map50:.4f}")
    print(f"mAP50-95 (Mask): {val.seg.map:.4f}")
    print(f"Precision:       {val.box.mp:.4f}")
    print(f"Recall:          {val.box.mr:.4f}")

    print("\nBenchmark CPU (50 inferencias)...")
    cpu = benchmark(MODEL_PATH, test_imgs, "cpu")
    print("Benchmark GPU (50 inferencias)...")
    gpu = benchmark(MODEL_PATH, test_imgs, 0)

    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\n{'='*54}")
    print("    COMPARACION GPU vs CPU -- YOLOv11n-seg (dataset propio)")
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*54}")
    print(f"{'Metrica':<26} {'CPU':>12} {'GPU':>12}")
    print(f"{'-'*54}")
    print(f"{'FPS':<26} {cpu['fps']:>12.1f} {gpu['fps']:>12.1f}")
    print(f"{'Tiempo/img (ms)':<26} {cpu['ms']:>12.1f} {gpu['ms']:>12.1f}")
    print(f"{'RAM (GB)':<26} {cpu['ram']:>12.2f} {gpu['ram']:>12.2f}")
    print(f"{'VRAM usada (GB)':<26} {'N/A':>12} {vram_used:>12.2f}")
    print(f"{'VRAM total (GB)':<26} {'N/A':>12} {vram_total:>12.2f}")
    print(f"{'-'*54}")
    print(f"{'Aceleracion GPU/CPU':<26} {gpu['fps']/cpu['fps']:>12.1f}x")
    print(f"{'='*54}")
