#include "ascon.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Define RATE based on the variant
#ifdef ASCON_AEAD_RATE
#define RATE ASCON_AEAD_RATE
#else
#define RATE 8
#endif

// Helper functions
__host__ __device__ void ascon_permutation(ascon_state_t *s, int rounds);

// Ascon permutation function
__host__ __device__ void ascon_permutation(ascon_state_t *s, int rounds)
{
    static const uint8_t RC[12] = {0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4};
    for (int r = 12 - rounds; r < 12; ++r)
    {
        s->x[2] ^= RC[r];
        s->x[0] ^= s->x[4];
        s->x[4] ^= s->x[3];
        s->x[2] ^= s->x[1];
        uint64_t T[5];
        for (int i = 0; i < 5; ++i)
        {
            T[i] = s->x[i] ^ (~s->x[(i + 1) % 5] & s->x[(i + 2) % 5]);
        }
        for (int i = 0; i < 5; ++i)
        {
            s->x[i] = T[i];
        }
        s->x[0] ^= s->x[4];
        s->x[4] ^= s->x[3];
        s->x[2] ^= s->x[1];
        s->x[0] = (s->x[0] >> 19) ^ (s->x[0] << (64 - 19)) ^ (s->x[0] >> 28) ^ (s->x[0] << (64 - 28));
        s->x[1] = (s->x[1] >> 61) ^ (s->x[1] << (64 - 61)) ^ (s->x[1] >> 39) ^ (s->x[1] << (64 - 39));
        s->x[2] = (s->x[2] >> 1) ^ (s->x[2] << (64 - 1)) ^ (s->x[2] >> 6) ^ (s->x[2] << (64 - 6));
        s->x[3] = (s->x[3] >> 10) ^ (s->x[3] << (64 - 10)) ^ (s->x[3] >> 17) ^ (s->x[3] << (64 - 17));
        s->x[4] = (s->x[4] >> 7) ^ (s->x[4] << (64 - 7)) ^ (s->x[4] >> 41) ^ (s->x[4] << (64 - 41));
    }
}

// AEAD functions
__host__ __device__ void ascon_loadkey(ascon_key_t *key, const uint8_t *k)
{
    memcpy(key->b, k, CRYPTO_KEYBYTES);
}

__host__ __device__ void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub)
{
    memset(s, 0, sizeof(ascon_state_t));
    s->x[0] = 0x80400c0600000000ULL ^ ((uint64_t)CRYPTO_KEYBYTES << 56) ^ ((uint64_t)ASCON_AEAD_RATE << 48);
    s->x[1] = key->x[0];
    s->x[2] = key->x[1];
    s->x[3] = ((uint64_t *)npub)[0];
    s->x[4] = ((uint64_t *)npub)[1];
    ascon_permutation(s, 12);
    s->x[3] ^= key->x[0];
    s->x[4] ^= key->x[1];
}

__host__ __device__ void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen)
{
    while (adlen >= ASCON_AEAD_RATE)
    {
        s->x[0] ^= ((uint64_t *)ad)[0];
        ascon_permutation(s, 6);
        ad += ASCON_AEAD_RATE;
        adlen -= ASCON_AEAD_RATE;
    }
    uint8_t lastblock[ASCON_AEAD_RATE] = {0};
    memcpy(lastblock, ad, adlen);
    lastblock[adlen] = 0x80;
    s->x[0] ^= ((uint64_t *)lastblock)[0];
    ascon_permutation(s, 6);
    s->x[4] ^= 1;
}

__host__ __device__ void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen)
{
    while (mlen >= ASCON_AEAD_RATE)
    {
        s->x[0] ^= ((uint64_t *)m)[0];
        ((uint64_t *)c)[0] = s->x[0];
        ascon_permutation(s, 6);
        m += ASCON_AEAD_RATE;
        c += ASCON_AEAD_RATE;
        mlen -= ASCON_AEAD_RATE;
    }
    uint8_t lastblock[ASCON_AEAD_RATE] = {0};
    memcpy(lastblock, m, mlen);
    lastblock[mlen] = 0x80;
    s->x[0] ^= ((uint64_t *)lastblock)[0];
    memcpy(c, &s->x[0], mlen);
}

__host__ __device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
{
    s->x[1] ^= k->x[0];
    s->x[2] ^= k->x[1];
    ascon_permutation(s, 12);
    s->x[3] ^= k->x[0];
    s->x[4] ^= k->x[1];
}

__global__ void ascon_aead_encrypt_kernel(uint8_t *t, uint8_t *c, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = mlen / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = (idx == gridDim.x * blockDim.x - 1) ? mlen : start + chunk_size;

    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);

    for (int i = start; i < end; i += ASCON_AEAD_RATE)
    {
        s->x[0] ^= ((uint64_t *)(m + i))[0];
        ((uint64_t *)(c + i))[0] = s->x[0];
        ascon_permutation(s, 6);
    }

    if (idx == gridDim.x * blockDim.x - 1)
    {
        int remaining = mlen % ASCON_AEAD_RATE;
        if (remaining > 0)
        {
            uint8_t lastblock[ASCON_AEAD_RATE] = {0};
            memcpy(lastblock, m + end, remaining);
            lastblock[remaining] = 0x80;
            s->x[0] ^= ((uint64_t *)lastblock)[0];
            memcpy(c + end, &s->x[0], remaining);
        }
    }

    ascon_final(&s, &key);
    if (idx == 0)
    {
        memcpy(t, &s.x[3], 16);
    }
}

int main()
{
    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (auto &byte : key)
        byte = dis(gen);
    for (auto &byte : nonce)
        byte = dis(gen);

    std::vector<uint8_t> plaintext = { /* your plaintext data */ };
    size_t plaintext_len = plaintext.size();
    std::vector<uint8_t> ciphertext(plaintext_len + 16);
    std::vector<uint8_t> tag(16);

    uint8_t *d_plaintext, *d_ciphertext, *d_tag, *d_nonce, *d_key;
    cudaMalloc(&d_plaintext, plaintext.size());
    cudaMalloc(&d_ciphertext, ciphertext.size());
    cudaMalloc(&d_tag, tag.size());
    cudaMalloc(&d_nonce, nonce.size());
    cudaMalloc(&d_key, key.size());

    cudaMemcpy(d_plaintext, plaintext.data(), plaintext.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce.data(), nonce.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice);

    auto start_encrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_encrypt_kernel<<<1, 1>>>(d_tag, d_ciphertext, d_plaintext, plaintext_len, nullptr, 0, d_nonce, d_key);
    cudaDeviceSynchronize();
    auto end_encrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
    std::cout << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;

    cudaMemcpy(tag.data(), d_tag, tag.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ciphertext.data(), d_ciphertext, ciphertext.size(), cudaMemcpyDeviceToHost);

    // Print or save the ciphertext and tag as needed

    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_tag);
    cudaFree(d_nonce);
    cudaFree(d_key);

    return 0;
}
