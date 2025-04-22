#include <omp.h> 
#include <iostream> 
#include <cmath>
#include <chrono>

using namespace std; 

double f(double x) {
    double result = 0;
    for(int i = 0; i < 10000000; i++) {
        result += sin(x);
    }
    return result;
}

int main() { 
    int a[100], b[100]; 
    // Инициализация массива b 
    for(int i = 0; i<100; i++) 
        b[i] = i;
        
    // --- Начало замера времени ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // Директива OpenMP для распараллеливания цикла 
    #pragma omp parallel for 
    for(int i = 0; i<100; i++) 
    { 
        a[i] = f(b[i]); 
        b[i] = 2*a[i]; 
    } 
    int result = 0; 
    // Далее значения a[i] и b[i] используются, например, так: 
    #pragma omp parallel for reduction(+ : result) 
    for(int i = 0; i<100; i++) 
        result += (a[i] + b[i]);
    // --- Конец замера времени ---
    auto end_time = std::chrono::high_resolution_clock::now();

    // Вычисление длительности
    std::chrono::duration<double> duration = end_time - start_time;
    cout << "Result = " << result << endl;
    cout << "Execution time: " << duration.count() << " seconds" << endl; // Вывод времени 
 
    return 0; 
}