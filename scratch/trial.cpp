#include <iostream>
// #include <omp.h>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>


int main() {
    const int num_threads = 8;
    const int total_iterations = 1000;
    double resource = 0;
    std::mutex resource_mutex;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&resource, &resource_mutex, total_iterations]() {
            int local_count = 0;

            // Perform floating point arithmetic
            for (int i = 0; i < total_iterations; ++i) {
                // Perform floating point operations here
                resource_mutex.lock();
                double result = i * 3.14159 + 2.71828;
                resource += result;
                local_count++;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                resource_mutex.unlock();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "Final resource value: " << resource << std::endl;

    return 0;
}