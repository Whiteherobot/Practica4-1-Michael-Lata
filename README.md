# Práctica 4-1: Visión por Computador con GPU

**Universidad Politécnica Salesiana — Visión por Computador**
**Período Lectivo:** Marzo – Agosto 2026
**Docente:** Ing. Vladimir Robles Bykbaev
**Estudiante:** Michael Lata

> Estructura de referencia inspirada en el repo de Jordy Romero
> (`../ClasificacionObjetosYOLO/`), pero con dataset, código y evidencia propios —
> la guía prohíbe reusar imágenes o código de otro grupo.

## Estructura

```
Practica4-1 Michael Lata/
├── Parte1A/                              # YOLOv11-seg — segmentación de objetos de tránsito
│   ├── datasets_raw/                     # 3 datasets Roboflow originales (detección)
│   │   ├── traffic_cone/                 # nc=1, YA trae polígonos reales
│   │   ├── traffic_highway/              # nc=5, bbox
│   │   └── us_road_signs/                # nc=3, bbox
│   ├── dataset_traffic_objects/          # dataset fusionado en formato YOLO-seg (8 clases)
│   ├── models/                           # yolo11n-seg.pt (base) + best.pt (entrenado)
│   ├── merge_dataset.py                  # fusiona los 3 datasets, bbox -> polígono rectangular
│   ├── train.py                          # fine-tuning YOLOv11n-seg
│   ├── benchmark.py                      # validación + benchmark GPU vs CPU (dataset de test)
│   ├── webcam_live.py                    # segmentación en vivo por webcam GPU vs CPU con overlay
│   └── runs/                             # logs/pesos de entrenamiento (generado)
│
├── Parte1B/                              # YOLOv12 / YOLOv26 + RealPLKSR — GPU vs CPU
│   ├── models/                           # yolo12n.pt, yolo26n.pt, RealPLKSR_4x.pth
│   └── benchmark_live.py                 # detección o super-resolución en vivo con overlay
│                                          # (FPS, RAM, nvidia-smi, MAC) para grabar pantalla
│
└── Parte1C/                              # Pipeline OpenCV CUDA — C++
    ├── main.cpp                          # CPU vs GPU-only (Gaussian, morfología, Canny, ecualización)
    ├── CMakeLists.txt
    └── resultados/
```

## Dataset propio (Parte 1A)

Fusión de 3 datasets Roboflow distintos, todos formato YOLOv11:

| Fuente | Clases originales | Formato |
|---|---|---|
| `traffic_cone` | traffic-cone | segmentación (polígono real) |
| `traffic_highway_dataset` | sharp-turn, traffic-sign, turn-sign, uturn, warning | bbox |
| `US Road Signs` | regulatory, stop, warning | bbox |

`merge_dataset.py` remapea a 8 clases unificadas (`traffic-cone, sharp-turn,
traffic-sign, turn-sign, uturn, warning, regulatory, stop` — `warning` se
fusiona entre las dos fuentes que lo tenían) y convierte cada bounding box a un
polígono rectangular de 4 esquinas para poder entrenar YOLO11n-**seg** (no hay
máscaras reales para esas dos fuentes, es una técnica válida cuando no se
dispone de segmentación real).

## Ejecución

### Parte 1A — requiere grabación propia (segmentación por webcam)
```bash
cd Parte1A
python merge_dataset.py     # ya ejecutado, regenera dataset_traffic_objects/ si hace falta
python train.py             # fine-tuning (~40 épocas)
python benchmark.py         # métricas + FPS GPU vs CPU (dataset de test)
python webcam_live.py --device gpu --source 0   # segmentación en vivo por webcam, grabar pantalla
python webcam_live.py --device cpu --source 0
```

### Parte 1B — requiere grabación propia
```bash
cd Parte1B
python benchmark_live.py --task detect --model yolo12 --device gpu --source 0
python benchmark_live.py --task detect --model yolo12 --device cpu --source 0
python benchmark_live.py --task detect --model yolo26 --device gpu --source 0
python benchmark_live.py --task sr --device gpu --source 0
python benchmark_live.py --task sr --device cpu --source 0
```
Super resolución con **RealPLKSR** (2024, arXiv 2404.11848) en vez de Real-ESRGAN
(2021), para cumplir el requisito de la guía de usar una red de no más de 2 años.
El overlay en pantalla muestra FPS, RAM, `nvidia-smi` (uso/temperatura GPU) y la
MAC address del equipo — grabar pantalla mientras corre para la evidencia que
pide la guía. `--source` acepta `0` (webcam) o una ruta a video.

### Parte 1C — requiere `opencv-cuda` + CUDA Toolkit instalados
```bash
sudo pacman -S cuda opencv-cuda
cd Parte1C
cmake -B build && cmake --build build
./build/practica4_1c imagen.jpg    # benchmark CPU vs GPU-only sobre una imagen
./build/practica4_1c 0             # en vivo con webcam
```

## Requisitos
- venv `cv-ai311` (Python 3.11.9, PyTorch 2.11+cu128, Ultralytics, OpenCV, spandrel, psutil)
- GPU NVIDIA con CUDA (RTX 4060 Laptop, 8 GB VRAM)
- Para Parte1C: CMake, OpenCV5 con módulos CUDA (`opencv-cuda`), CUDA Toolkit
