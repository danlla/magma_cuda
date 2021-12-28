#pragma once
#include "magma.cuh"

class magma_gpu :public magma
{
public:
    void encrypt(block* buf, size_t n) const override;
    void decrypt(block* buf, size_t n) const override;
    magma_gpu(const std::array<unsigned int, 8>& key);
};