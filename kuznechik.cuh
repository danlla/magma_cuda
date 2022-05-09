#pragma once
#include <memory>
#include <array>
#include "cuda_runtime.h"
class kuznechik
{
public:
	union block
	{
		unsigned char c[16];
		unsigned int uint[4];
		unsigned long long ull[2];
	};

protected:
	unsigned int keys[8];
	__host__ void encrypt_block(block& src) const;
	__host__ void decrypt_block(block& src) const;
public:
	kuznechik(const std::array<unsigned int, 8>& key);
	virtual void encrypt(block* buf, size_t n) const;
	virtual void decrypt(block* buf, size_t n) const;
};