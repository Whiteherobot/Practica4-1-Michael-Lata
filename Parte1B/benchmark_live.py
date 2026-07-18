"""Práctica 4-1, Parte 1B: benchmark GPU vs CPU en vivo.

Muestra en pantalla (para grabar) FPS, uso de RAM, uso de VRAM/nvidia-smi y la MAC
address del equipo, mientras corre:
  - detección de objetos con YOLOv12 o YOLOv26 sobre webcam/video, o
  - super resolución con RealPLKSR (2024, arXiv 2404.11848) sobre webcam/video.
    Se eligió RealPLKSR en vez de Real-ESRGAN (2021) porque la guía exige una red
    de super resolución de no más de 2 años de existencia.

Uso:
    python benchmark_live.py --task detect --model yolo12 --device gpu --source 0
    python benchmark_live.py --task detect --model yolo26 --device cpu --source 0
    python benchmark_live.py --task sr --device gpu --source 0
    python benchmark_live.py --task sr --device cpu --source ruta/a/video.mp4

Controles: Q para salir. El overlay queda "quemado" en lo que se ve en pantalla,
listo para grabar con OBS / grabador de pantalla del sistema.
"""
import argparse
import os
import subprocess
import time
import uuid

import cv2
import numpy as np
import psutil
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "yolo12": os.path.join(HERE, "models", "yolo12n.pt"),
    "yolo26": os.path.join(HERE, "models", "yolo26n.pt"),
}
SR_WEIGHTS = os.path.join(HERE, "models", "RealPLKSR_4x.pth")


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


def draw_overlay(frame, device_name, fps, mac, extra_line=""):
    h, w = frame.shape[:2]
    color = (0, 255, 0) if device_name == "GPU" else (0, 0, 255)
    ram_gb = psutil.virtual_memory().used / 1024**3
    lines = [
        f"{device_name} | FPS: {fps:.1f}",
        f"RAM: {ram_gb:.2f} GB",
        nvidia_smi_summary(),
        f"MAC: {mac}",
    ]
    if extra_line:
        lines.append(extra_line)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        y += 26
    return frame


def run_detect(model_name, device, source):
    from ultralytics import YOLO

    model = YOLO(MODELS[model_name])
    device_arg = 0 if device == "gpu" else "cpu"
    device_name = "GPU" if device == "gpu" else "CPU"
    mac = get_mac_address()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

    win = f"Practica4-1B | {model_name} | {device_name}"
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
        out = draw_overlay(out, device_name, fps, mac, extra_line=f"Modelo: {model_name}")
        cv2.imshow(win, out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if fps_hist:
        print(f"FPS promedio ({device_name}, {model_name}): {np.mean(fps_hist):.2f}")


def run_sr(device, source):
    from spandrel import ModelLoader

    torch_device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
    model = ModelLoader().load_from_file(SR_WEIGHTS).to(torch_device).eval()
    device_name = "GPU" if torch_device.type == "cuda" else "CPU"
    mac = get_mac_address()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

    win = f"Practica4-1B | RealPLKSR x4 | {device_name}"
    fps_hist = []
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            t0 = time.perf_counter()
            img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch_device)
            sr_tensor = model(tensor)
            sr = sr_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            sr = cv2.cvtColor((sr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            fps = 1 / max(time.perf_counter() - t0, 1e-6)
            fps_hist.append(fps)
            out = draw_overlay(sr, device_name, fps, mac, extra_line="RealPLKSR x4 (2024, entrada reducida a 1/2)")
            cv2.imshow(win, out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    if fps_hist:
        print(f"FPS promedio ({device_name}, RealPLKSR): {np.mean(fps_hist):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "sr"], required=True)
    ap.add_argument("--model", choices=["yolo12", "yolo26"], default="yolo12")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--source", default="0", help="0 = webcam, o ruta a un archivo de video")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    if args.task == "detect":
        run_detect(args.model, args.device, source)
    else:
        run_sr(args.device, source)
