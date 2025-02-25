#include "ascon.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// Define RATE based on the variant
#ifdef ASCON_AEAD_RATE
#define RATE ASCON_AEAD_RATE
#else
#define RATE 8
#endif

#define NUM_FILES 24

using namespace std;

// Helper functions
__device__ uint64_t warp_shuffle(uint64_t val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__constant__ uint8_t RC[12] = {0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4};
__constant__ uint64_t SBOX[5] = {0x80400c0600000000ULL, 0x1ULL, 0x2ULL, 0x3ULL, 0x4ULL}; // Example S-box values

__device__ void ascon_permutation(ascon_state_t *s, int rounds) {
    // Use registers for x0, x1, x2, x3, x4
    register uint64_t x0 = s->x[0];
    register uint64_t x1 = s->x[1];
    register uint64_t x2 = s->x[2];
    register uint64_t x3 = s->x[3];
    register uint64_t x4 = s->x[4];

    #pragma unroll
    for (int r = 12 - rounds; r < 12; ++r) {
        x2 ^= RC[r];
        x0 ^= x4;
        x4 ^= x3;
        x2 ^= x1;

        uint64_t t0 = x0 ^ (~x1 & x2);
        uint64_t t1 = x1 ^ (~x2 & x3);
        uint64_t t2 = x2 ^ (~x3 & x4);
        uint64_t t3 = x3 ^ (~x4 & x0);
        uint64_t t4 = x4 ^ (~x0 & x1);

        x0 = t0;
        x1 = t1;
        x2 = t2;
        x3 = t3;
        x4 = t4;

        x0 ^= x4;
        x4 ^= x3;
        x2 ^= x1;

        x0 = (x0 >> 19) ^ (x0 << (64 - 19)) ^ (x0 >> 28) ^ (x0 << (64 - 28));
        x1 = (x1 >> 61) ^ (x1 << (64 - 61)) ^ (x1 >> 39) ^ (x1 << (64 - 39));
        x2 = (x2 >> 1) ^ (x2 << (64 - 1)) ^ (x2 >> 6) ^ (x2 << (64 - 6));
        x3 = (x3 >> 10) ^ (x3 << (64 - 10)) ^ (x3 >> 17) ^ (x3 << (64 - 17));
        x4 = (x4 >> 7) ^ (x4 << (64 - 7)) ^ (x4 >> 41) ^ (x4 << (64 - 41));
    }

    s->x[0] = x0;
    s->x[1] = x1;
    s->x[2] = x2;
    s->x[3] = x3;
    s->x[4] = x4;
}

// AEAD functions
__device__ void ascon_loadkey(ascon_key_t *key, const uint8_t *k)
{
    memcpy(key->b, k, CRYPTO_KEYBYTES);
}

__device__ void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub)
{
    memset(s, 0, sizeof(ascon_state_t));
    s->x[0] = 0x80400c0600000000ULL ^ ((uint64_t)CRYPTO_KEYBYTES << 56) ^ ((uint64_t)ASCON_AEAD_RATE << 48);
    memcpy(&s->x[1], key->b, 16);
    memcpy(&s->x[3], npub, 16);
    ascon_permutation(s, 12);
    s->x[3] ^= key->x[0];
    s->x[4] ^= key->x[1];
}

__device__ void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen)
{
    while (adlen >= ASCON_AEAD_RATE)
    {
        uint64_t block;
        memcpy(&block, ad, ASCON_AEAD_RATE);
        s->x[0] ^= block;
        ascon_permutation(s, 6);
        ad += ASCON_AEAD_RATE;
        adlen -= ASCON_AEAD_RATE;
    }
    uint8_t lastblock[ASCON_AEAD_RATE] = {0};
    memcpy(lastblock, ad, adlen);
    lastblock[adlen] = 0x80;
    uint64_t block;
    memcpy(&block, lastblock, ASCON_AEAD_RATE);
    s->x[0] ^= block;
    ascon_permutation(s, 6);
    s->x[4] ^= 1;
}

__device__ void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen)
{
    while (mlen >= ASCON_AEAD_RATE)
    {
        uint64_t block;
        memcpy(&block, m, ASCON_AEAD_RATE);
        s->x[0] ^= block;
        memcpy(c, &s->x[0], ASCON_AEAD_RATE);
        ascon_permutation(s, 6);
        m += ASCON_AEAD_RATE;
        c += ASCON_AEAD_RATE;
        mlen -= ASCON_AEAD_RATE;
    }
    uint8_t lastblock[ASCON_AEAD_RATE] = {0};
    memcpy(lastblock, m, mlen);
    lastblock[mlen] = 0x80;
    uint64_t block;
    memcpy(&block, lastblock, ASCON_AEAD_RATE);
    s->x[0] ^= block;
    memcpy(c, &s->x[0], mlen);
}

__device__ void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen)
{
    while (clen >= ASCON_AEAD_RATE)
    {
        uint64_t cblock;
        memcpy(&cblock, c, ASCON_AEAD_RATE);
        uint64_t mblock = s->x[0] ^ cblock;
        memcpy(m, &mblock, ASCON_AEAD_RATE);
        s->x[0] = cblock;
        ascon_permutation(s, 6);
        c += ASCON_AEAD_RATE;
        m += ASCON_AEAD_RATE;
        clen -= ASCON_AEAD_RATE;
    }
    uint8_t lastblock[ASCON_AEAD_RATE] = {0};
    memcpy(lastblock, c, clen);
    lastblock[clen] = 0x80;
    uint64_t cblock;
    memcpy(&cblock, lastblock, ASCON_AEAD_RATE);
    uint64_t mblock = s->x[0] ^ cblock;
    memcpy(m, &mblock, clen);
    s->x[0] = cblock;
}

__device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
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

__global__ void ascon_aead_encrypt_kernel(uint8_t *t, uint8_t *c, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k, int num_files) {
    int file_idx = blockIdx.x;
    if (file_idx >= num_files) return;

    int idx = threadIdx.x;
    int chunk_size = mlen / blockDim.x;
    int start = file_idx * mlen + idx * chunk_size;
    int end = (idx == blockDim.x - 1) ? (file_idx + 1) * mlen : start + chunk_size;

    __shared__ ascon_state_t shared_s;
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k + file_idx * 16);
    ascon_initaead(&s, &key, npub + file_idx * 16);
    ascon_adata(&s, ad, adlen);

    if (threadIdx.x == 0) {
        shared_s = s;
    }
    __syncthreads();

    register uint64_t x0 = shared_s.x[0];
    register uint64_t x1 = shared_s.x[1];
    register uint64_t x2 = shared_s.x[2];
    register uint64_t x3 = shared_s.x[3];
    register uint64_t x4 = shared_s.x[4];

    for (int i = start; i < end; i += ASCON_AEAD_RATE) {
        uint64_t block;
        memcpy(&block, m + i, ASCON_AEAD_RATE);
        x0 ^= block;
        memcpy(c + i, &x0, ASCON_AEAD_RATE);
        ascon_permutation(&shared_s, 6);
    }

    if (idx == blockDim.x - 1) {
        int remaining = mlen % ASCON_AEAD_RATE;
        if (remaining > 0) {
            uint8_t lastblock[ASCON_AEAD_RATE] = {0};
            memcpy(lastblock, m + end, remaining);
            lastblock[remaining] = 0x80;
            uint64_t block;
            memcpy(&block, lastblock, ASCON_AEAD_RATE);
            x0 ^= block;
            memcpy(c + end, &x0, remaining);
        }
    }

    ascon_final(&shared_s, &key);
    if (idx == 0) {
        memcpy(t + file_idx * 16, &shared_s.x[3], 16);
    }
}

__global__ void ascon_aead_decrypt_kernel(uint8_t *m, const uint8_t *t, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k, int *result, int num_files) {
    int file_idx = blockIdx.x;
    if (file_idx >= num_files) return;

    int idx = threadIdx.x;
    int chunk_size = clen / blockDim.x;
    int start = file_idx * clen + idx * chunk_size;
    int end = (idx == blockDim.x - 1) ? (file_idx + 1) * clen : start + chunk_size;

    __shared__ ascon_state_t shared_s;
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k + file_idx * 16);
    ascon_initaead(&s, &key, npub + file_idx * 16);
    ascon_adata(&s, ad, adlen);

    if (threadIdx.x == 0) {
        shared_s = s;
    }
    __syncthreads();

    register uint64_t x0 = shared_s.x[0];
    register uint64_t x1 = shared_s.x[1];
    register uint64_t x2 = shared_s.x[2];
    register uint64_t x3 = shared_s.x[3];
    register uint64_t x4 = shared_s.x[4];

    for (int i = start; i < end; i += ASCON_AEAD_RATE) {
        uint64_t cblock;
        memcpy(&cblock, c + i, ASCON_AEAD_RATE);
        uint64_t mblock = x0 ^ cblock;
        memcpy(m + i, &mblock, ASCON_AEAD_RATE);
        x0 = cblock;
        ascon_permutation(&shared_s, 6);
    }

    if (idx == blockDim.x - 1) {
        int remaining = clen % ASCON_AEAD_RATE;
        if (remaining > 0) {
            uint8_t lastblock[ASCON_AEAD_RATE] = {0};
            memcpy(lastblock, c + end, remaining);
            lastblock[remaining] = 0x80;
            uint64_t cblock;
            memcpy(&cblock, lastblock, ASCON_AEAD_RATE);
            uint64_t mblock = x0 ^ cblock;
            memcpy(m + end, &mblock, remaining);
            x0 = cblock;
        }
    }

    ascon_final(&shared_s, &key);
    if (idx == 0) {
        result[file_idx] = ascon_compare(t + file_idx * 16, (uint8_t *)&shared_s.x[3], 16);
    }
}

// Helper function to convert hex string to byte array
void hex_to_bytes(const std::string &hex, std::vector<uint8_t> &bytes)
{
    bytes.resize((hex.length() + 1) / 2); // Ensure 8-byte alignment
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

// Function to write byte array to hex string file
void write_bytes_to_hex_file(const std::string &filename, const std::vector<uint8_t> &bytes)
{
    std::ofstream file(filename);
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        file << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
    }
    file.close();
}

int main()
{
    std::vector<std::vector<uint8_t>> plaintexts(NUM_FILES);
    std::vector<std::vector<uint8_t>> ciphertexts(NUM_FILES);
    std::vector<std::vector<uint8_t>> decrypted(NUM_FILES);
    std::vector<uint8_t> tags(NUM_FILES * 16);
    std::vector<uint8_t> keys(NUM_FILES * 16);
    std::vector<uint8_t> nonces(NUM_FILES * 16);
    std::vector<int> results(NUM_FILES);

    for (int i = 0; i < NUM_FILES; ++i) {
        std::string hex_string = read_hex_from_file("frame_" + std::to_string(i) + ".txt");
        hex_to_bytes(hex_string, plaintexts[i]);
        ciphertexts[i].resize((plaintexts[i].size() + 15) / 16 * 16); // Ensure 8-byte alignment
        decrypted[i].resize((plaintexts[i].size() + 15) / 16 * 16); // Ensure 8-byte alignment
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < NUM_FILES; ++i) {
        for (int j = 0; j < 16; ++j) {
            keys[i * 16 + j] = dis(gen);
            nonces[i * 16 + j] = dis(gen);
        }
    }

    uint8_t *d_plaintexts, *d_ciphertexts, *d_tags, *d_nonces, *d_keys;
    int *d_results;
    cudaError_t err;

    size_t total_plaintext_size = 0;
    for (const auto& pt : plaintexts) {
        total_plaintext_size += (pt.size() + 15) / 16 * 16;
    }

    err = cudaMalloc(&d_plaintexts, total_plaintext_size);
    if (err != cudaSuccess) {
        printf("CUDA Malloc Error (d_plaintexts): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_ciphertexts, total_plaintext_size);
    if (err != cudaSuccess) {
        printf("CUDA Malloc Error (d_ciphertexts): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_tags, tags.size());
    if (err != cudaSuccess) {
        printf("CUDA Malloc Error (d_tags): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_nonces, nonces.size());
    if (err != cudaSuccess) {
        printf("CUDA Malloc Error (d_nonces): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_keys, keys.size());
    if (err != cudaSuccess) {
        printf("CUDA Malloc Error (d_keys): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_results, sizeof(int) * NUM_FILES);
    if (err != cudaSuccess) {
        printf("CUDA Malloc Error (d_results): %s\n", cudaGetErrorString(err));
    }

    size_t offset = 0;
    for (int i = 0; i < NUM_FILES; ++i) {
        err = cudaMemcpy(d_plaintexts + offset, plaintexts[i].data(), plaintexts[i].size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA Memcpy Error (plaintexts[%d] → d_plaintexts): %s\n", i, cudaGetErrorString(err));
        }
        offset += (plaintexts[i].size() + 15) / 16 * 16;
    }
    err = cudaMemcpy(d_nonces, nonces.data(), nonces.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy Error (nonces → d_nonces): %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_keys, keys.data(), keys.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy Error (keys → d_keys): %s\n", cudaGetErrorString(err));
    }

    int threadsPerBlock = 256;
    int numBlocks = NUM_FILES;

    printf("Kernel launch config: numBlocks=%d, threadsPerBlock=%d\n", numBlocks, threadsPerBlock);

    auto start_encrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_encrypt_kernel<<<numBlocks, threadsPerBlock>>>(d_tags, d_ciphertexts, d_plaintexts, plaintexts[0].size(), nullptr, 0, d_nonces, d_keys, NUM_FILES);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    auto end_encrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
    std::cout << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;

    err = cudaMemcpy(tags.data(), d_tags, tags.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy Error (d_tags → tags): %s\n", cudaGetErrorString(err));
    }
    offset = 0;
    for (int i = 0; i < NUM_FILES; ++i) {
        err = cudaMemcpy(ciphertexts[i].data(), d_ciphertexts + offset, ciphertexts[i].size(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA Memcpy Error (d_ciphertexts → ciphertexts[%d]): %s\n", i, cudaGetErrorString(err));
        }
        offset += (ciphertexts[i].size() + 15) / 16 * 16;
    }

    auto start_decrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_decrypt_kernel<<<numBlocks, threadsPerBlock>>>(d_plaintexts, d_tags, d_ciphertexts, plaintexts[0].size() + 16, nullptr, 0, d_nonces, d_keys, d_results, NUM_FILES);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    auto end_decrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_decrypt = end_decrypt - start_decrypt;
    std::cout << "Decryption time: " << elapsed_decrypt.count() << " seconds" << std::endl;

    offset = 0;
    for (int i = 0; i < NUM_FILES; ++i) {
        err = cudaMemcpy(decrypted[i].data(), d_plaintexts + offset, decrypted[i].size(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA Memcpy Error (d_plaintexts → decrypted[%d]): %s\n", i, cudaGetErrorString(err));
        }
        offset += (decrypted[i].size() + 15) / 16 * 16;
    }
    err = cudaMemcpy(results.data(), d_results, sizeof(int) * NUM_FILES, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy Error (d_results → results): %s\n", cudaGetErrorString(err));
    }

    for (int i = 0; i < NUM_FILES; ++i) {
        std::cout << "File " << i << " encryption result:" << std::endl;
        for (size_t j = 0; j < ciphertexts[i].size(); ++j) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)ciphertexts[i][j];
        }
        std::cout << std::endl;

        std::cout << "File " << i << " decryption result:" << std::endl;
        for (size_t j = 0; j < decrypted[i].size(); ++j) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)decrypted[i][j];
        }
        std::cout << std::endl;

        std::cout << "File " << i << " tag:" << std::endl;
        for (int j = 0; j < 16; ++j) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)tags[i * 16 + j];
        }
        std::cout << std::endl;

        std::cout << "File " << i << " verification result: " << (results[i] == 0 ? "Success" : "Failure") << std::endl;
    }

    for (int i = 0; i < NUM_FILES; ++i) {
        write_bytes_to_hex_file("encrypt_" + std::to_string(i) + ".txt", ciphertexts[i]);
        write_bytes_to_hex_file("plaintext_" + std::to_string(i) + ".txt", decrypted[i]);
    }

    cudaFree(d_plaintexts);
    cudaFree(d_ciphertexts);
    cudaFree(d_tags);
    cudaFree(d_nonces);
    cudaFree(d_keys);
    cudaFree(d_results);

    return 0;
}
