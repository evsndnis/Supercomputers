#include <omp.h>
#include <iostream>
using namespace std;

int main() {
    int sum = 0; // Эта переменная будет разделяемой по умолчанию
    const int a_size = 100;
    int a[a_size], id, size;

    for(int i = 0; i < a_size; i++)
        a[i] = i;

    // Директива OpenMP parallel без reduction, но с private(id, size)
    #pragma omp parallel private(id, size)
    { // Начало параллельной области
        id = omp_get_thread_num();
        size = omp_get_num_threads();

        // Объявляем локальную переменную для частичной суммы в каждом потоке
        int local_sum = 0;

        // Разделяем работу между потоками
        int integer_part = a_size / size;
        int remainder = a_size % size;
        int a_local_size = integer_part +
                           ((id < remainder) ? 1 : 0);
        int start = integer_part * id +
                    ((id < remainder) ? id : remainder);
        int end = start + a_local_size;

        // Каждый поток суммирует элементы
        // своей части массива в local_sum
        for(int i = start; i < end; i++)
            local_sum += a[i];

        // Используем critical секцию для атомарного обновления общей суммы
        // Только один поток может войти в этот блок одновременно
        #pragma omp critical
        {
            sum += local_sum; // Обновляем общую сумму
        }

        // Каждый поток выводит свою частичную сумму (теперь local_sum)
        cout << "Thread " << id << ", partial sum = " << local_sum << endl;
    } // Конец параллельной области

    // sum содержит корректную общую сумму
    cout << "Final sum = " << sum << endl;

    return 0;
}