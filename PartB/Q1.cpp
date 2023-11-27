#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

int main() {
    int K_values[] = {1, 5, 10, 50, 100};

    for (int i = 0; i < 5; i++) {
        int K = K_values[i];
        int *array1 = new int[K * 1000000];
        int *array2 = new int[K * 1000000];
        int *result = new int[K * 1000000];

        // Initialize arrays with random values
        for (int j = 0; j < K * 1000000; j++) {
            array1[j] = rand();
            array2[j] = rand();
        }

        // Profile execution time
        auto start_time = chrono::high_resolution_clock::now();

        // Add elements of arrays
        for (int j = 0; j < K * 1000000; j++) {
            result[j] = array1[j] + array2[j];
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        cout << "Time to execute for K = " << K << " million: " << duration.count() << " ms" << endl;

        // Free memory
        delete[] array1;
        delete[] array2;
        delete[] result;
    }

    return 0;
}
