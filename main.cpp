#include <iostream>
#include <chrono>
#include <random>
#include "magma.cuh"
#include "magma_gpu.cuh"
#include <fstream>
#include "argparse.hpp"

void bench(const magma& m, size_t n, const std::unique_ptr<magma::block[]>& message);

void encrypt_file(const magma& m, const char* input_file, const char* output_file, size_t buf_size);

void decrypt_file(const magma& m, const char* input_file, const char* output_file, size_t buf_size);

int main(int argc, char* argv[])
{


    for (size_t i = 0; i < argc; ++i)
        if (!(strcmp(argv[i],"-b") && strcmp(argv[i], "--bench")))
        {
            std::cout << "Testing: 128mb" << std::endl;
            std::array<unsigned int, 8> keys = { 0xccddeeff, 0x8899aabb, 0x44556677, 0x00112233, 0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc };
            size_t n = 128 * 1024*1024 / sizeof(magma::block);
            auto message = std::make_unique<magma::block[]>(n);
            std::mt19937 eng;
            eng.seed(15);
            for (size_t i = 0; i < n; ++i)
            {
                message[i].uint[0] = eng();
                message[i].uint[1] = eng();
            }

            std::cout << "CPU" << std::endl;
            magma m1(keys);
            bench(m1, n, message);

            std::cout << "GPU" << std::endl;
            magma_gpu m2(keys);
            bench(m2, n, message);
            return 0;
        }


    argparse::ArgumentParser program("magma", "1.0.0");
    program.add_argument("-e", "--encrypt")
        .help("encrypt data")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-d", "--decrypt")
        .help("decrypt data")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-b", "--bench")
        .help("testing program. test gpu and cpu 128mb data. for using use only this argument")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-r", "--random")
        .help("file for saving random key");
   /* program.add_argument("-p", "--password")
        .help("password key");*/
    program.add_argument("-k", "--key")
        .help("for encrypt read a key from file\n\t\tfor decrypt read a key from file");
    program.add_argument("input file")
        .help("input file");
    program.add_argument("output file")
        .help("output file");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::array<unsigned int, 8> keys;
    
    if (program["--decrypt"] == false && program["--encrypt"] == false)
    {
        std::cout << "You need choose encrypt or decrypt";
        exit(EXIT_FAILURE);
    }
    /*if (program.present("-p").has_value())
    {
        if (program.present("-r").has_value())
        {
            std::cout << "Only password or random";
            exit(EXIT_FAILURE);
        }
        std::cout << program.present("-p").value() << std::endl;
    }*/
    else if (program.present("-r").has_value())
    {
        if (program["--decrypt"] == true)
        {
            std::cout << "you can't use random to decrypt";
            exit(EXIT_FAILURE);
        }
        std::cout << "random" << std::endl;
        std::mt19937 engine;
        engine.seed(1);
        for (int i = 0; i < 8; ++i)
            keys[i] = engine();
        std::ofstream ofile(program.present("-r").value(), std::ios::binary);
        ofile.setf(std::ios_base::hex);
        ofile.write((char*)keys.data(), 64);
    }
    else
    {
        if(!program.present("-k").has_value())
        {
            std::cout << "you need use random or key";
            exit(EXIT_FAILURE);
        }
        std::ifstream ifile(program.present("-k").value(), std::ios::binary);
        if (!ifile)
        {
            std::cout << "empty key file";
            exit(EXIT_FAILURE);
        }
        ifile.read((char*)keys.data(),64);
    }

    if (program["--encrypt"] == true) {
        if (program["--decrypt"] == true)
        {
            std::cout << "Only encrypt or decrypt";
            exit(EXIT_FAILURE);
        }
        std::cout << "encrypt";
        std::cout << " from " << program.get<std::string>("input file") << " to " << program.get<std::string>("output file") << std::endl;
        magma m(keys);
        encrypt_file(m, program.get<std::string>("input file").c_str(), program.get<std::string>("output file").c_str(), 500);
    }
    if (program["--decrypt"] == true) {
        std::cout << "decrypt";
        std::cout << " from " << program.get<std::string>("input file") << " to " << program.get<std::string>("output file") << std::endl;
        magma m(keys);
        decrypt_file(m, program.get<std::string>("input file").c_str(), program.get<std::string>("output file").c_str(), 500);
    }

}

int addition(magma::block* buf, size_t byte)
{
    if (byte % 8 == 0)
    {
        for (int i = 0; i < 8; ++i)
            buf[byte / 8].c[i] = 8;
    }
    else
    {
        int dif = 8 - byte % 8;
        for (int i = 8 - dif; i < 8; ++i)
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
    magma::block* buf = new magma::block[buf_size + 1];
    size_t count = 0;
    while (true)
    {
        ifile.read((char*)buf, buf_size * sizeof(magma::block));
        count = ifile.gcount();
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
    magma::block* buf = new magma::block[buf_size + 1];
    size_t count = 0;
    while (true)
    {
        ifile.read((char*)buf, buf_size * sizeof(magma::block));
        count = ifile.gcount();
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
    std::cout << "works :)" << std::endl;
    std::cout << std::dec << time << "ms" << std::endl;
}