#pragma once
#include "magma.cuh"

class magma_gpu :public magma
{
private:
    friend __global__ void encrypt_kernel(block*, size_t, magma_gpu);
public:
    void encrypt(block* buf, size_t n) const override;
    magma_gpu(const std::array<unsigned int, 8>& key);
};