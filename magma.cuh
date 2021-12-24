#pragma once
#include <memory>
#include <vector>
#include <array>
#include "cuda_runtime.h"
class magma
{
public:
	union block
	{
		unsigned char c[8];
		unsigned int uint[2];
		unsigned long long ull;
	};

protected:
	unsigned int keys[8];
	__host__ void encrypt_block(block& src) const; //had to rename it  - otherwise the device linker failed
	__host__ void decrypt_block(block& src) const;
public:
	magma(const std::array<unsigned int, 8>& key);
	virtual void encrypt(block* buf, size_t n) const;
	virtual void decrypt(block* buf, size_t n) const;
};
