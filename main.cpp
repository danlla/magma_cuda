#include <iostream>
#include <chrono>
#include <random>
#include "magma.cuh"
#include "magma_gpu.cuh"
#include <fstream>

void bench(const magma& m, size_t n, const std::unique_ptr<magma::block[]>& message);

int main()
{
    std::array<unsigned int, 8> keys = { 0xccddeeff, 0x8899aabb, 0x44556677, 0x00112233, 0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc };
    magma m(keys);
    magma::block test;
    test.ull = 0x1032547698badcfe;


    std::mt19937 engine;
    engine.seed(1);
    size_t n = 128 / sizeof(magma::block);//64*1024*1024;
    auto message = std::make_unique<magma::block[]>(n);
    for (size_t i = 0; i < n; ++i)
    {
        message[i].uint[0] = engine();
        message[i].uint[1] = engine();
    }
    //bench(m, n, message);

    magma_gpu m2(keys);
    bench(m2, n, message);

    /*magma::block* buf = new magma::block[500];
    std::ifstream file("F:\\magma\\test.txt");
    if (!file.is_open())
        std::cout << "error";
    else
    {
        size_t i = 0;
        while (!file.eof()&&i<500)
        {
            file >> buf[i].ull;
            ++i;
        }
        std::cout << buf->ull;
    }*/

}

/*one task - one function*/
void bench(const magma& m, size_t n, const std::unique_ptr<magma::block[]>& message) {
    auto tmp = std::make_unique<magma::block[]>(n);
    std::copy_n(message.get(), n, tmp.get());
    auto start = std::chrono::high_resolution_clock::now();
    m.encrypt(message.get(), n);
    auto end = std::chrono::high_resolution_clock::now();
    m.decrypt(message.get(), n);


#pragma omp parallel for //faster compare
    for (int64_t i = 0; i < n; ++i)
    {
        if (message[i].ull != tmp[i].ull)
        {
            std::cout << "doesn't work :c";
            exit(EXIT_FAILURE);
        }
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Works :)" << std::endl;
    std::cout << std::dec << time << "ms" << std::endl;
}
