#include "kuznechik.cuh"
#include <memory>
#include <stdio.h>


constexpr unsigned char L1[16][16] =
{
};

constexpr unsigned char L2[16][16] =
{
};

void kuznechik::encrypt_block(block& src2) const
{
	
}

void kuznechik::decrypt_block(block& src) const
{
	
}

kuznechik::kuznechik(const std::array<unsigned int, 8>& key)
{
	for (size_t i = 0; i < 8; ++i)
	{
		keys[i] = key[i];
	}
}

void kuznechik::encrypt(block* buf, size_t n) const
{
#pragma omp parallel for
	for (int64_t i = 0; i < n; ++i)
		encrypt_block(buf[i]);
}

void kuznechik::decrypt(block* buf, size_t n) const
{
#pragma omp parallel for
	for (int64_t i = 0; i < n; ++i)
		decrypt_block(buf[i]);
}
