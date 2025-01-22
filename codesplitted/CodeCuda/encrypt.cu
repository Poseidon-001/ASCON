#include "ascon.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

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

__host__ __device__ void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen)
{
    while (clen >= ASCON_AEAD_RATE)
    {
        uint64_t cblock = ((uint64_t *)c)[0];
        ((uint64_t *)m)[0] = s->x[0] ^ cblock;
        s->x[0] = cblock;
        ascon_permutation(s, 6);
        c += ASCON_AEAD_RATE;
        m += ASCON_AEAD_RATE;
        clen -= ASCON_AEAD_RATE;
    }
    uint8_t lastblock[ASCON_AEAD_RATE] = {0};
    memcpy(lastblock, c, clen);
    lastblock[clen] = 0x80;
    uint64_t cblock = ((uint64_t *)lastblock)[0];
    ((uint64_t *)m)[0] = s->x[0] ^ cblock;
    s->x[0] = cblock;
}

__host__ __device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
{
    s->x[1] ^= k->x[0];
    s->x[2] ^= k->x[1];
    ascon_permutation(s, 12);
    s->x[3] ^= k->x[0];
    s->x[4] ^= k->x[1];
}

__device__ int ascon_compare(const uint8_t *a, const uint8_t *b, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        if (a[i] != b[i])
        {
            return -1;
        }
    }
    return 0;
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

__global__ void ascon_aead_decrypt_kernel(uint8_t *m, const uint8_t *t, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k, int *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = clen / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = (idx == gridDim.x * blockDim.x - 1) ? clen : start + chunk_size;

    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);

    for (int i = start; i < end; i += ASCON_AEAD_RATE)
    {
        uint64_t cblock = ((uint64_t *)(c + i))[0];
        ((uint64_t *)(m + i))[0] = s->x[0] ^ cblock;
        s->x[0] = cblock;
        ascon_permutation(s, 6);
    }

    if (idx == gridDim.x * blockDim.x - 1)
    {
        int remaining = clen % ASCON_AEAD_RATE;
        if (remaining > 0)
        {
            uint8_t lastblock[ASCON_AEAD_RATE] = {0};
            memcpy(lastblock, c + end, remaining);
            lastblock[remaining] = 0x80;
            uint64_t cblock = ((uint64_t *)lastblock)[0];
            ((uint64_t *)(m + end))[0] = s->x[0] ^ cblock;
            s->x[0] = cblock;
        }
    }

    ascon_final(&s, &key);
    if (idx == 0)
    {
        *result = ascon_compare(t, (uint8_t *)&s.x[3], 16);
    }
}

// Helper function to convert hex string to byte array
void hex_to_bytes(const std::string &hex, std::vector<uint8_t> &bytes)
{
    bytes.resize(hex.length() / 2);
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        std::stringstream ss;
        ss << std::hex << hex.substr(2 * i, 2);
        int byte;
        ss >> byte;
        bytes[i] = static_cast<uint8_t>(byte);
    }
}

// Function to read hex string from file
std::string read_hex_from_file(const std::string &filename)
{
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main()
{
    // Example data for encryption and decryption
    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
    std::vector<uint8_t> ad = {0x09, 0x0A, 0x0B, 0x0C};
    std::vector<uint8_t> plaintext;
    std::vector<uint8_t> ciphertext;
    std::vector<uint8_t> tag(16);
    std::vector<uint8_t> decrypted;

    // Read plaintext from file
    std::string hex_string = read_hex_from_file("output_hex.txt");
    hex_to_bytes(hex_string, plaintext);
    size_t plaintext_len = plaintext.size();
    ciphertext.resize(plaintext_len + 16);
    decrypted.resize(plaintext_len);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (auto &byte : key)
        byte = dis(gen);
    for (auto &byte : nonce)
        byte = dis(gen);

    uint8_t *d_plaintext, *d_ciphertext, *d_tag, *d_nonce, *d_key;
    int *d_result;
    cudaMalloc(&d_plaintext, plaintext.size());
    cudaMalloc(&d_ciphertext, ciphertext.size());
    cudaMalloc(&d_tag, tag.size());
    cudaMalloc(&d_nonce, nonce.size());
    cudaMalloc(&d_key, key.size());
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_plaintext, plaintext.data(), plaintext.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce.data(), nonce.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice);

    // Measure encryption time
    auto start_encrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_encrypt_kernel<<<1, 1>>>(d_tag, d_ciphertext, d_plaintext, plaintext_len, ad.data(), ad.size(), d_nonce, d_key);
    cudaDeviceSynchronize();
    auto end_encrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
    std::cout << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;

    cudaMemcpy(tag.data(), d_tag, tag.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ciphertext.data(), d_ciphertext, ciphertext.size(), cudaMemcpyDeviceToHost);

    // Write ciphertext to file
    std::ofstream output_file("ciphertext_encrypt.txt");
    for (auto byte : ciphertext)
    {
        output_file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    output_file << endl;
    output_file.close();

    // Measure decryption time
    auto start_decrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_decrypt_kernel<<<1, 1>>>(d_plaintext, d_tag, d_ciphertext, plaintext_len + 16, ad.data(), ad.size(), d_nonce, d_key, d_result);
    cudaDeviceSynchronize();
    auto end_decrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_decrypt = end_decrypt - start_decrypt;
    std::cout << "Decryption time: " << elapsed_decrypt.count() << " seconds" << std::endl;

    cudaMemcpy(decrypted.data(), d_plaintext, decrypted.size(), cudaMemcpyDeviceToHost);
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Decryption result: " << (result == 0 ? "Success" : "Failure") << std::endl;

    // Write decrypted plaintext to file
    if (result == 0)
    {
        std::ofstream output_file("plaintext.txt");
        for (auto byte : decrypted)
        {
            output_file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
        }
        output_file << endl;
        output_file.close();
    }

    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_tag);
    cudaFree(d_nonce);
    cudaFree(d_key);
    cudaFree(d_result);

    return 0;
}
