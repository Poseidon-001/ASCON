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
    s->x[1] = key->x[0];
    s->x[2] = key->x[1];
    s->x[3] = ((uint64_t *)npub)[0];
    s->x[4] = ((uint64_t *)npub)[1];
    ascon_permutation(s, 12);
    s->x[3] ^= key->x[0];
    s->x[4] ^= key->x[1];
}

__device__ void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen)
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

__device__ void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen)
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

__device__ void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen)
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

__global__ void ascon_aead_encrypt_kernel(uint8_t *t, uint8_t *c, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k) {
    // Increase the number of blocks and threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = mlen / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = (idx == gridDim.x * blockDim.x - 1) ? mlen : start + chunk_size;

    __shared__ ascon_state_t shared_s;
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);

    // Copy state to shared memory
    if (threadIdx.x == 0) {
        shared_s = s;
    }
    __syncthreads();

    // Use registers for x0, x1, x2, x3, x4
    register uint64_t x0 = shared_s.x[0];
    register uint64_t x1 = shared_s.x[1];
    register uint64_t x2 = shared_s.x[2];
    register uint64_t x3 = shared_s.x[3];
    register uint64_t x4 = shared_s.x[4];

    for (int i = start; i < end; i += ASCON_AEAD_RATE) {
        x0 ^= ((uint64_t *)(m + i))[0];
        ((uint64_t *)(c + i))[0] = x0;
        ascon_permutation(&shared_s, 6);
    }

    if (idx == gridDim.x * blockDim.x - 1) {
        int remaining = mlen % ASCON_AEAD_RATE;
        if (remaining > 0) {
            uint8_t lastblock[ASCON_AEAD_RATE] = {0};
            memcpy(lastblock, m + end, remaining);
            lastblock[remaining] = 0x80;
            x0 ^= ((uint64_t *)lastblock)[0];
            memcpy(c + end, &x0, remaining);
        }
    }

    ascon_final(&shared_s, &key);
    if (idx == 0) {
        memcpy(t, &shared_s.x[3], 16);
    }
}

__global__ void ascon_aead_decrypt_kernel(uint8_t *m, const uint8_t *t, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k, int *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        printf("Decryption kernel started on GPU!\n");
    }

    // Increase the number of blocks and threads
    int chunk_size = clen / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = (idx == gridDim.x * blockDim.x - 1) ? clen : start + chunk_size;

    __shared__ ascon_state_t shared_s;
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);

    // Copy state to shared memory
    if (threadIdx.x == 0) {
        shared_s = s;
    }
    __syncthreads();

    // Use registers for x0, x1, x2, x3, x4
    register uint64_t x0 = shared_s.x[0];
    register uint64_t x1 = shared_s.x[1];
    register uint64_t x2 = shared_s.x[2];
    register uint64_t x3 = shared_s.x[3];
    register uint64_t x4 = shared_s.x[4];

    for (int i = start; i < end; i += ASCON_AEAD_RATE) {
        uint64_t cblock = ((uint64_t *)(c + i))[0];
        ((uint64_t *)(m + i))[0] = x0 ^ cblock;
        x0 = cblock;
        ascon_permutation(&shared_s, 6);
    }

    if (idx == gridDim.x * blockDim.x - 1) {
        int remaining = clen % ASCON_AEAD_RATE;
        if (remaining > 0) {
            uint8_t lastblock[ASCON_AEAD_RATE] = {0};
            memcpy(lastblock, c + end, remaining);
            lastblock[remaining] = 0x80;
            uint64_t cblock = ((uint64_t *)lastblock)[0];
            ((uint64_t *)(m + end))[0] = x0 ^ cblock;
            x0 = cblock;
        }
    }

    ascon_final(&shared_s, &key);
    if (idx == 0) {
        *result = ascon_compare(t, (uint8_t *)&shared_s.x[3], 16);
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
    // Read hex string from file
    std::string hex_string = read_hex_from_file("frame_0.txt");
    std::vector<uint8_t> plaintext;
    hex_to_bytes(hex_string, plaintext);

    size_t plaintext_len = plaintext.size();
    std::vector<uint8_t> ciphertext(plaintext_len + 16);
    std::vector<uint8_t> tag(16);
    std::vector<uint8_t> decrypted(plaintext_len);

    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
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

    // Debugging: Copy the first 16 bytes of plaintext from GPU to host and print them
    uint8_t debug_plaintext[16];
    cudaMemcpy(debug_plaintext, d_plaintext, 16, cudaMemcpyDeviceToHost);
    printf("First 16 bytes of plaintext on GPU:\n");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", debug_plaintext[i]);
    }
    printf("\n");

    // Increase the number of blocks and threads
    int threadsPerBlock = 256;
    int numBlocks = (plaintext_len + threadsPerBlock - 1) / threadsPerBlock;

    printf("Kernel launch config: numBlocks=%d, threadsPerBlock=%d\n", numBlocks, threadsPerBlock);

    // Measure encryption time
    auto start_encrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_encrypt_kernel<<<numBlocks, threadsPerBlock>>>(d_tag, d_ciphertext, d_plaintext, plaintext_len, nullptr, 0, d_nonce, d_key);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    auto end_encrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
    std::cout << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;

    // Debugging: Copy the first 16 bytes of ciphertext from GPU to host and print them
    uint8_t debug_ciphertext[16];
    cudaMemcpy(debug_ciphertext, d_ciphertext, 16, cudaMemcpyDeviceToHost);
    printf("First 16 bytes of encrypted data (ciphertext):\n");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", debug_ciphertext[i]);
    }
    printf("\n");

    cudaMemcpy(tag.data(), d_tag, tag.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ciphertext.data(), d_ciphertext, ciphertext.size(), cudaMemcpyDeviceToHost);

    // Measure decryption time
    auto start_decrypt = std::chrono::high_resolution_clock::now();
    ascon_aead_decrypt_kernel<<<numBlocks, threadsPerBlock>>>(d_plaintext, d_tag, d_ciphertext, plaintext_len + 16, nullptr, 0, d_nonce, d_key, d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    auto end_decrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_decrypt = end_decrypt - start_decrypt;
    std::cout << "Decryption time: " << elapsed_decrypt.count() << " seconds" << std::endl;

    // Debugging: Copy the first 16 bytes of decrypted data from GPU to host and print them
    cudaMemcpy(debug_plaintext, d_plaintext, 16, cudaMemcpyDeviceToHost);
    printf("First 16 bytes of decrypted data on GPU:\n");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", debug_plaintext[i]);
    }
    printf("\n");

    cudaMemcpy(decrypted.data(), d_plaintext, decrypted.size(), cudaMemcpyDeviceToHost);
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Encryption result:" << std::endl;
    for (size_t i = 0; i < ciphertext.size(); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)ciphertext[i];
    }
    std::cout << std::endl;

    std::cout << "Decryption result:" << std::endl;
    for (size_t i = 0; i < decrypted.size(); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)decrypted[i];
    }
    std::cout << std::endl;

    std::cout << "Tag:" << std::endl;
    for (size_t i = 0; i < tag.size(); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)tag[i];
    }
    std::cout << std::endl;

    std::cout << "Verification result: " << (result == 0 ? "Success" : "Failure") << std::endl;

    // Write results to files
    write_bytes_to_hex_file("encrypt.txt", ciphertext);
    write_bytes_to_hex_file("plaintext.txt", decrypted);    

    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_tag);
    cudaFree(d_nonce);
    cudaFree(d_key);
    cudaFree(d_result);

    return 0;
}
