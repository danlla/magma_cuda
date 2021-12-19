#include <iostream>
#include <chrono>
#include <random>
#include "magma.cuh"
#include "magma_gpu.cuh"
#include <fstream>

void bench(const magma& m, size_t n, const std::unique_ptr<magma::block[]>& message);

void encrypt_file(const magma& m, const char* input_file, const char* output_file, size_t buf_size);

void decrypt_file(const magma& m, const char* input_file, const char* output_file, size_t buf_size);

int main()
{
    std::array<unsigned int, 8> keys = { 0xccddeeff, 0x8899aabb, 0x44556677, 0x00112233, 0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc };
    magma m(keys);
    magma::block test;
    test.ull = 0x1032547698badcfe;


    std::mt19937 engine;
    engine.seed(1);
    size_t n = 128*1024 / sizeof(magma::block);//64*1024*1024;
    auto message = std::make_unique<magma::block[]>(n);
    for (size_t i = 0; i < n; ++i)
    {
        message[i].uint[0] = engine();
        message[i].uint[1] = engine();
    }
    //bench(m, n, message);

    magma_gpu m2(keys);
    //bench(m2, n, message);

    encrypt_file(m, "test.txt","encrypted.txt",500);
    decrypt_file(m, "encrypted.txt", "decrypted.txt",500);
}

int addition(magma::block* buf, size_t byte)
{
    if (byte % 8 == 0)
    {
        //buf[byte / 8].ull = 0x0808080808080808;
        for (int i = 0; i < 8; ++i)
            buf[byte / 8].c[i] = 8;
    }
    else
    {
        int dif = 8-byte % 8;
        for (int i = 8-dif; i < 8; ++i)
            buf[byte / 8].c[i] = dif;
    }
    return 1;
}

int cut_addition(const magma::block& b)
{
    return b.c[7];
}

void encrypt_file(const magma& m, const char* input_file, const char* output_file, size_t buf_size)
{
    std::ifstream ifile(input_file, std::ios::binary);
    std::ofstream ofile(output_file, std::ios::binary);
    magma::block* buf = new magma::block[buf_size+1];
    size_t count = 0;
    while (true)
    {
        ifile.read((char*)buf, buf_size * sizeof(magma::block));
        count = ifile.gcount();
        std::cout << "read " << count << std::endl;
        if (!ifile)
            break;
        m.encrypt(buf, count / 8);
        ofile.write((char*)buf, (count / 8) * sizeof(magma::block));
    }
    int add = addition(buf, count);
    m.encrypt(buf, count / 8 + add);
    ofile.write((char*)buf, (count / 8 + add) * sizeof(magma::block));
    delete[] buf;
}

void decrypt_file(const magma& m, const char* input_file, const char* output_file, size_t buf_size)
{
    std::ifstream ifile(input_file, std::ios::binary);
    std::ofstream ofile(output_file, std::ios::binary);
    magma::block* buf = new magma::block[buf_size+1];
    size_t count = 0;
    while(true)
    {
        ifile.read((char*)buf, buf_size * sizeof(magma::block));
        count = ifile.gcount();
        std::cout << "read " << count << std::endl;
        if (!ifile)
            break;
        m.decrypt(buf, count / 8);
        ofile.write((char*)buf, (count / 8) * sizeof(magma::block));
    }
    int extra = 0;
    m.decrypt(buf, count / 8);
    extra = cut_addition(buf[count / 8 - 1]);
    ofile.write((char*)buf, (count / 8) * sizeof(magma::block) - extra);
    delete[] buf;
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
