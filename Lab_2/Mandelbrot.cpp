#include <omp.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <string>
#include <cstdlib>  // Для atoi()

const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_ITER = 1000;

int mandelbrot(double x, double y) {
    std::complex<double> point(x, y);
    std::complex<double> z(0, 0);
    int iter = 0;
    while (abs(z) < 2 && iter < MAX_ITER) {
        z = z * z + point;
        iter++;
    }
    return iter;
}

void compute_mandelbrot(int* image, const std::string& schedule_type, int chunk_size) {
    #pragma omp parallel
    #pragma omp single
    {
        for (int y_start = 0; y_start < HEIGHT; y_start += chunk_size) {
            int y_end = std::min(y_start + chunk_size, HEIGHT);
            #pragma omp task firstprivate(y_start, y_end)
            {
                for (int y = y_start; y < y_end; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        double real = -2.0 + (x * 3.0) / WIDTH;
                        double imag = -1.5 + (y * 3.0) / HEIGHT;
                        image[y * WIDTH + x] = mandelbrot(real, imag);
                    }
                }
            }
        }
        #pragma omp taskwait
    }
}

int main(int argc, char** argv) {
    // Параметры по умолчанию
    std::string schedule_type = "static";
    int chunk_size = 10;

    // Обработка аргументов
    if (argc > 1) schedule_type = argv[1];
    if (argc > 2) chunk_size = std::atoi(argv[2]);
    if (chunk_size < 1) chunk_size = 1;  // Защита от некорректных значений

    int* image = new int[WIDTH * HEIGHT];

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();
    compute_mandelbrot(image, schedule_type, chunk_size);
    auto end = std::chrono::high_resolution_clock::now();

    // Сохранение в файл
    std::ofstream file("mandelbrot.pgm");
    file << "P2\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        file << (image[i] % 255) << " ";
    }
    file.close();

    // Вывод результатов
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Schedule: " << schedule_type 
              << " | Chunk: " << chunk_size 
              << " | Time: " << duration.count() << " ms" 
              << " | Threads: " << omp_get_max_threads() 
              << std::endl;

    delete[] image;
    return 0;
}
