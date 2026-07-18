"""Practica 4-1, Parte 1A: segmentacion en vivo por webcam, GPU vs CPU.

La guia pide segmentar objetos de imagenes capturadas por webcam, comparando
rendimiento GPU vs CPU. benchmark.py ya mide FPS/mAP sobre el dataset de test;
este script corre el modelo entrenado (models/best.pt) en vivo sobre la webcam
con el mismo overlay de FPS/RAM/nvidia-smi usado en Parte1B, listo para grabar.

Uso:
    python webcam_live.py --device gpu --source 0
    python webcam_live.py --device cpu --source 0

Controles: Q para salir.
"""
import argparse
import os
import subprocess
import time
import uuid

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "best.pt")


def get_mac_address():
    mac = uuid.getnode()
    return ":".join(f"{(mac >> ele) & 0xFF:02x}" for ele in range(40, -8, -8))


def nvidia_smi_summary():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        ).decode().strip()
        util, mem_used, mem_total, temp = [x.strip() for x in out.split(",")]
        return f"GPU util: {util}%  VRAM: {mem_used}/{mem_total} MiB  Temp: {temp}C"
    except Exception:
        return "nvidia-smi no disponible"


def draw_overlay(frame, device_name, fps, mac):
    color = (0, 255, 0) if device_name == "GPU" else (0, 0, 255)
    ram_gb = psutil.virtual_memory().used / 1024**3
    lines = [
        f"{device_name} | FPS: {fps:.1f}",
        f"RAM: {ram_gb:.2f} GB",
        nvidia_smi_summary(),
        f"MAC: {mac}",
        "YOLO11n-seg (dataset propio, 8 clases)",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        y += 26
    return frame


def run(device, source):
    model = YOLO(MODEL_PATH)
    device_arg = 0 if device == "gpu" else "cpu"
    device_name = "GPU" if device == "gpu" else "CPU"
    mac = get_mac_address()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

    win = f"Practica4-1A | YOLO11n-seg | {device_name}"
    fps_hist = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model(frame, device=device_arg, conf=0.35, verbose=False)
        fps = 1 / max(time.perf_counter() - t0, 1e-6)
        fps_hist.append(fps)
        out = res[0].plot()
        out = draw_overlay(out, device_name, fps, mac)
        cv2.imshow(win, out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if fps_hist:
        print(f"FPS promedio ({device_name}): {np.mean(fps_hist):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--source", default="0", help="0 = webcam, o ruta a un archivo de video")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(args.device, source)
