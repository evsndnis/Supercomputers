#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <string>

using namespace std;
using namespace std::chrono;

vector<double> jacobi_method(const vector<vector<double>>& A, 
                            const vector<double>& b, 
                            double eps, 
                            int max_iter,
                            int num_threads,
                            const string& schedule_type,
                            int chunk_size) {
    int n = b.size();
    vector<double> x_old(n, 0.0);
    vector<double> x_new(n);
    
    int iter = 0;
    double norm;
    
    omp_set_num_threads(num_threads);
    
    auto start_time = high_resolution_clock::now();
    
    do {
        norm = 0.0;
        
        // Директива parallel for с выбранным schedule
        if (schedule_type == "static") {
            #pragma omp parallel for reduction(max:norm) schedule(static, chunk_size)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) sum += A[i][j] * x_old[j];
                }
                x_new[i] = (b[i] - sum) / A[i][i];
                double diff = fabs(x_new[i] - x_old[i]);
                if (diff > norm) norm = diff;
            }
        }
        else if (schedule_type == "dynamic") {
            #pragma omp parallel for reduction(max:norm) schedule(dynamic, chunk_size)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) sum += A[i][j] * x_old[j];
                }
                x_new[i] = (b[i] - sum) / A[i][i];
                double diff = fabs(x_new[i] - x_old[i]);
                if (diff > norm) norm = diff;
            }
        }
        else if (schedule_type == "guided") {
            #pragma omp parallel for reduction(max:norm) schedule(guided, chunk_size)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) sum += A[i][j] * x_old[j];
                }
                x_new[i] = (b[i] - sum) / A[i][i];
                double diff = fabs(x_new[i] - x_old[i]);
                if (diff > norm) norm = diff;
            }
        }
        else {
            #pragma omp parallel for reduction(max:norm)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) sum += A[i][j] * x_old[j];
                }
                x_new[i] = (b[i] - sum) / A[i][i];
                double diff = fabs(x_new[i] - x_old[i]);
                if (diff > norm) norm = diff;
            }
        }
        
        x_old = x_new;
        iter++;
        
    } while (norm > eps && iter < max_iter);
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    double seconds = duration.count() / 1000000.0;
    
    cout << "Num of iterations: " << iter << endl;
    cout << "Norm: " << scientific << setprecision(2) << norm << endl;
    cout << "Time: " << fixed << setprecision(2) << seconds << " sec" << endl;
    
    return x_new;
}

void print_help() {
    cout << "Use: ./Jacobi_enhabced [size] [accuracy] [max_ietrs] [threads] [schedule] [chunk_size]\n";
    cout << "  schedule: static, dynamic, guided (as default: static)\n";
    cout << "  chunk_size: size of block for schedule (as default: 32)\n";
    cout << "Example: ./Jacobi_enhanced 1000 1e-6 1000 4 dynamic 16\n";
}

int main(int argc, char* argv[]) {
    // Параметры по умолчанию
    int n = 1000;
    double eps = 1e-6;
    int max_iter = 1000;
    int num_threads = omp_get_max_threads();
    string schedule_type = "static";
    int chunk_size = 32;
    
    // Обработка аргументов командной строки
    if (argc > 1) {
        if (string(argv[1]) == "--help") {
            print_help();
            return 0;
        }
        n = atoi(argv[1]);
        if (argc > 2) eps = atof(argv[2]);
        if (argc > 3) max_iter = atoi(argv[3]);
        if (argc > 4) num_threads = atoi(argv[4]);
        if (argc > 5) schedule_type = argv[5];
        if (argc > 6) chunk_size = atoi(argv[6]);
    }
    
    // Проверка корректности параметров schedule
    if (schedule_type != "static" && schedule_type != "dynamic" && schedule_type != "guided") {
        cerr << "Error: wrong type of schedule. Choose: static, dynamic, guided\n";
        return 1;
    }
    
    cout << "Parameters:\n";
    cout << "  Size of system: " << n << "x" << n << endl;
    cout << "  Accuracy: " << eps << endl;
    cout << "  Max num of iterations: " << max_iter << endl;
    cout << "  Threads: " << num_threads << endl;
    cout << "  Schedule: " << schedule_type << " with chunk_size=" << chunk_size << endl;
    
    // Инициализация матрицы и вектора
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);
    
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) A[i][j] = n + 1;  // Диагональное преобладание
            else A[i][j] = 1.0;
        }
        b[i] = 2.0 * i + 1;
    }
    
    // Решение системы
    vector<double> x = jacobi_method(A, b, eps, max_iter, num_threads, schedule_type, chunk_size);
    
    // Вывод части результатов
    cout << "\nFirst 5 components:\n";
    for (int i = 0; i < 5; i++) {
        cout << "x[" << i << "] = " << x[i] << endl;
    }
    
    return 0;
}
