"""Benchmark offline (no interactivo) GPU vs CPU para Parte1B: YOLOv12/YOLOv26 y RealPLKSR.

Complementa a benchmark_live.py (pensado para grabar pantalla en vivo con overlay
de MAC/nvidia-smi): este script corre sobre imagenes fijas y sirve para tener
numeros reproducibles para el informe, sin depender de la grabacion en vivo.
"""
import glob
import os
import time

import numpy as np
import psutil
import torch
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "yolo12": os.path.join(HERE, "models", "yolo12n.pt"),
    "yolo26": os.path.join(HERE, "models", "yolo26n.pt"),
}
SR_WEIGHTS = os.path.join(HERE, "models", "RealPLKSR_4x.pth")


def benchmark_detect(model_name, images, device, n=30):
    from ultralytics import YOLO
    m = YOLO(MODELS[model_name])
    for img in images[:3]:
        m(img, device=device, verbose=False)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        m(images[i % len(images)], device=device, verbose=False)
        times.append(time.perf_counter() - t0)
    return {"fps": 1 / np.mean(times), "ms": np.mean(times) * 1000}


def benchmark_sr(frame, device, n=30):
    from spandrel import ModelLoader
    torch_device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
    model = ModelLoader().load_from_file(SR_WEIGHTS).to(torch_device).eval()
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch_device)
    with torch.no_grad():
        for _ in range(3):
            model(tensor)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(tensor)
            times.append(time.perf_counter() - t0)
    return {"fps": 1 / np.mean(times), "ms": np.mean(times) * 1000}


if __name__ == "__main__":
    sample_dir = os.path.join(HERE, "..", "Parte1A", "dataset_traffic_objects", "test", "images")
    imgs = glob.glob(os.path.join(sample_dir, "*.jpg"))[:10]
    if not imgs:
        raise RuntimeError("No hay imagenes de muestra disponibles")

    print(f"{'='*60}\n YOLOv12n / YOLOv26n -- deteccion GPU vs CPU\n{'='*60}")
    print(f"{'Modelo':<10} {'Device':<6} {'FPS':>10} {'ms/img':>10}")
    for name in ["yolo12", "yolo26"]:
        cpu = benchmark_detect(name, imgs, "cpu")
        gpu = benchmark_detect(name, imgs, 0)
        print(f"{name:<10} {'CPU':<6} {cpu['fps']:>10.1f} {cpu['ms']:>10.1f}")
        print(f"{name:<10} {'GPU':<6} {gpu['fps']:>10.1f} {gpu['ms']:>10.1f}")
        print(f"{'':<10} {'accel':<6} {gpu['fps']/cpu['fps']:>9.1f}x")

    print(f"\n{'='*60}\n RealPLKSR x4 -- super resolucion GPU vs CPU\n{'='*60}")
    frame = cv2.imread(imgs[0])
    frame = cv2.resize(frame, (320, 240))
    sr_cpu = benchmark_sr(frame, "cpu")
    sr_gpu = benchmark_sr(frame, "gpu")
    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{'Device':<6} {'FPS':>10} {'ms/img':>10}")
    print(f"{'CPU':<6} {sr_cpu['fps']:>10.2f} {sr_cpu['ms']:>10.1f}")
    print(f"{'GPU':<6} {sr_gpu['fps']:>10.2f} {sr_gpu['ms']:>10.1f}")
    print(f"accel: {sr_gpu['fps']/sr_cpu['fps']:.1f}x")
    print(f"VRAM usada: {vram_used:.2f} GB / total: {vram_total:.2f} GB")
