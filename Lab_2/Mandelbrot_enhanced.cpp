#include <omp.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cstdlib>

const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_ITER = 1000;
long baseline_time = 0;

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

void compute_mandelbrot(int* image, int chunk_size) {
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

void test_matrix_combination() {
    const std::vector<int> thread_counts = {1, 2, 4, 8};
    const std::vector<int> chunk_sizes = {1, 10, 50, 100};
    
    std::cout << "\n=== Threads × Chunk Size Matrix ===\n";
    std::cout << "        ";
    for (int chunk : chunk_sizes) std::cout << std::setw(8) << "Ch=" << chunk;
    std::cout << "\n";
    
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        std::cout << "T=" << std::setw(2) << threads << " |";
        
        for (int chunk : chunk_sizes) {
            auto start = std::chrono::high_resolution_clock::now();
            int* image = new int[WIDTH * HEIGHT];
            compute_mandelbrot(image, chunk);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << std::setw(8) << duration.count() << " ";
            
            delete[] image;
        }
        std::cout << "\n";
    }
}

void save_image(int* image, const std::string& filename) {
    std::ofstream file(filename);
    file << "P2\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        file << (image[i] % 255) << " ";
    }
    file.close();
}

int main(int argc, char** argv) {
    // Установка базового времени (1 поток)
    omp_set_num_threads(1);
    int* baseline_image = new int[WIDTH * HEIGHT];
    auto baseline_start = std::chrono::high_resolution_clock::now();
    compute_mandelbrot(baseline_image, 10);
    auto baseline_end = std::chrono::high_resolution_clock::now();
    baseline_time = std::chrono::duration_cast<std::chrono::milliseconds>(baseline_end - baseline_start).count();
    save_image(baseline_image, "baseline.pgm");
    delete[] baseline_image;

    // Запуск комплексного теста
    test_matrix_combination();

    // Дополнительный тест с пользовательскими параметрами
    if (argc == 3) {
        int threads = atoi(argv[1]);
        int chunk_size = atoi(argv[2]);
        
        omp_set_num_threads(threads);
        int* custom_image = new int[WIDTH * HEIGHT];
        
        auto start = std::chrono::high_resolution_clock::now();
        compute_mandelbrot(custom_image, chunk_size);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "\nCustom test: Threads=" << threads 
                  << " Chunk=" << chunk_size
                  << " Time=" << duration.count() << " ms"
                  << " Speedup=" << std::fixed << std::setprecision(2) 
                  << (float)baseline_time/duration.count() << "x\n";
        
        save_image(custom_image, "custom.pgm");
        delete[] custom_image;
    }

    return 0;
}