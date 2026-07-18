// Practica 4-1, Parte 1C - Pipeline de preprocesamiento OpenCV: CPU vs GPU (CUDA) y GPU-only.
// Operaciones: suavizado (Gaussian), morfologia (erosion/dilatacion), Canny, ecualizacion de histograma.
//
// Uso:
//   ./practica4_1c imagen.jpg     -> procesa una imagen fija, guarda resultados y tiempos
//   ./practica4_1c 0              -> procesa video de webcam en vivo (device 0), overlay de FPS

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Pipeline CPU: cada operacion se aplica con cv::Mat estandar.
cv::Mat pipeline_cpu(const cv::Mat& frame, double& ms_total) {
    auto t0 = Clock::now();

    cv::Mat gray, blurred, eq, edges, morph;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
    cv::equalizeHist(blurred, eq);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(eq, morph, kernel);
    cv::dilate(morph, morph, kernel);
    cv::Canny(morph, edges, 50, 150);

    ms_total = elapsed_ms(t0);
    cv::Mat out;
    cv::cvtColor(edges, out, cv::COLOR_GRAY2BGR);
    return out;
}

// Pipeline GPU-only: un solo upload al inicio, todo el trabajo con cv::cuda::GpuMat,
// un solo download al final (evita el cuello de botella de transferencias CPU<->GPU).
cv::Mat pipeline_gpu_only(const cv::Mat& frame, double& ms_total) {
    auto t0 = Clock::now();

    cv::cuda::GpuMat d_frame, d_gray, d_blur, d_eq, d_morph, d_edges;
    d_frame.upload(frame);

    cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);

    auto gauss = cv::cuda::createGaussianFilter(d_gray.type(), d_gray.type(), cv::Size(5, 5), 1.5);
    gauss->apply(d_gray, d_blur);

    cv::cuda::equalizeHist(d_blur, d_eq);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    auto erode_f = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, d_eq.type(), kernel);
    auto dilate_f = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, d_eq.type(), kernel);
    erode_f->apply(d_eq, d_morph);
    dilate_f->apply(d_morph, d_morph);

    auto canny = cv::cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_morph, d_edges);

    cv::Mat result;
    d_edges.download(result);

    ms_total = elapsed_ms(t0);
    cv::Mat out;
    cv::cvtColor(result, out, cv::COLOR_GRAY2BGR);
    return out;
}

void run_on_image(const std::string& path) {
    cv::Mat frame = cv::imread(path);
    if (frame.empty()) {
        std::cerr << "No se pudo leer la imagen: " << path << std::endl;
        return;
    }

    const int N = 50;
    std::vector<double> t_cpu, t_gpu;
    cv::Mat out_cpu, out_gpu;
    double ms;

    for (int i = 0; i < N; ++i) {
        out_cpu = pipeline_cpu(frame, ms);
        t_cpu.push_back(ms);
    }
    for (int i = 0; i < N; ++i) {
        out_gpu = pipeline_gpu_only(frame, ms);
        t_gpu.push_back(ms);
    }

    double avg_cpu = std::accumulate(t_cpu.begin(), t_cpu.end(), 0.0) / N;
    double avg_gpu = std::accumulate(t_gpu.begin(), t_gpu.end(), 0.0) / N;

    std::cout << "==========================================\n";
    std::cout << " Comparacion CPU vs GPU-only (" << N << " iteraciones)\n";
    std::cout << "==========================================\n";
    std::cout << "CPU:  " << avg_cpu << " ms/frame  (" << 1000.0 / avg_cpu << " FPS)\n";
    std::cout << "GPU:  " << avg_gpu << " ms/frame  (" << 1000.0 / avg_gpu << " FPS)\n";
    std::cout << "Aceleracion GPU/CPU: " << avg_cpu / avg_gpu << "x\n";

    cv::imwrite("resultados/resultado_cpu.png", out_cpu);
    cv::imwrite("resultados/resultado_gpu.png", out_gpu);
    std::cout << "Resultados guardados en resultados/resultado_cpu.png y resultado_gpu.png\n";
}

void run_on_webcam(int device) {
    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir la camara " << device << std::endl;
        return;
    }
    std::cout << "Presiona 'g' para pipeline GPU-only, 'c' para CPU, 'q' para salir\n";

    bool use_gpu = true;
    cv::Mat frame, out;
    double ms;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        out = use_gpu ? pipeline_gpu_only(frame, ms) : pipeline_cpu(frame, ms);
        double fps = 1000.0 / ms;

        std::string label = (use_gpu ? "GPU-only" : "CPU") + std::string(" | FPS: ") + std::to_string(fps);
        cv::putText(out, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    use_gpu ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::imshow("Practica4-1C", out);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 'g') use_gpu = true;
        if (key == 'c') use_gpu = false;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagen.jpg | indice_camara>\n";
        return 1;
    }
    std::string arg = argv[1];
    bool is_number = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);

    if (is_number) {
        run_on_webcam(std::stoi(arg));
    } else {
        run_on_image(arg);
    }
    return 0;
}
