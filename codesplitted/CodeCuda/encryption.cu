#include "ascon.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// Helper functions
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

// Ascon functions
__device__ void ascon_permutation(ascon_state_t *s, int rounds)
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

__device__ void ascon_loadkey(ascon_key_t *key, const uint8_t *k)
{
    std::memcpy(key->b, k, CRYPTO_KEYBYTES);
}

__device__ void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub)
{
    std::memset(s, 0, sizeof(ascon_state_t));
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
    std::memcpy(lastblock, ad, adlen);
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
    std::memcpy(lastblock, m, mlen);
    lastblock[mlen] = 0x80;
    s->x[0] ^= ((uint64_t *)lastblock)[0];
    std::memcpy(c, &s->x[0], mlen);
}

__device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
{
    s->x[1] ^= k->x[0];
    s->x[2] ^= k->x[1];
    ascon_permutation(s, 12);
    s->x[3] ^= k->x[0];
    s->x[4] ^= k->x[1];
}

__device__ void ascon_aead_encrypt(uint8_t *t, uint8_t *c, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);
    ascon_encrypt(&s, c, m, mlen);
    ascon_final(&s, &key);
    std::memcpy(t, &s.x[3], 16);
}

// CUDA kernel for Ascon encryption
__global__ void ascon_encrypt_kernel(uint8_t *d_ciphertext, const uint8_t *d_plaintext, size_t plaintext_len, const uint8_t *d_nonce, const uint8_t *d_key)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < plaintext_len)
    {
        uint8_t *c = d_ciphertext + idx * (plaintext_len + 16);
        const uint8_t *m = d_plaintext + idx * plaintext_len;
        uint8_t t[16];
        ascon_aead_encrypt(t, c, m, plaintext_len, nullptr, 0, d_nonce, d_key);
    }
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();  // Bắt đầu đo thời gian

    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (auto &byte : key)
        byte = dis(gen);
    for (auto &byte : nonce)
        byte = dis(gen);

    std::ifstream input_file("output_hex.txt");
    std::ofstream output_file("ciphertext.txt");
    if (!input_file.is_open() || !output_file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        return -1;
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(input_file, line))
    {
        lines.push_back(line);
    }

    size_t num_lines = lines.size();
    size_t plaintext_len = lines[0].length() / 2;
    size_t ciphertext_len = plaintext_len + 16;

    uint8_t *h_plaintext = new uint8_t[num_lines * plaintext_len];
    uint8_t *h_ciphertext = new uint8_t[num_lines * ciphertext_len];

    for (size_t i = 0; i < num_lines; ++i)
    {
        std::vector<uint8_t> temp_plaintext;
        hex_to_bytes(lines[i], temp_plaintext);
        std::memcpy(h_plaintext + i * plaintext_len, temp_plaintext.data(), plaintext_len);
    }

    uint8_t *d_plaintext, *d_ciphertext, *d_nonce, *d_key;
    cudaMalloc(&d_plaintext, num_lines * plaintext_len * sizeof(uint8_t));
    cudaMalloc(&d_ciphertext, num_lines * ciphertext_len * sizeof(uint8_t));
    cudaMalloc(&d_nonce, 16 * sizeof(uint8_t));
    cudaMalloc(&d_key, 16 * sizeof(uint8_t));

    cudaMemcpy(d_plaintext, h_plaintext, num_lines * plaintext_len * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce.data(), 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key.data(), 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_lines + blockSize - 1) / blockSize;
    ascon_encrypt_kernel<<<numBlocks, blockSize>>>(d_ciphertext, d_plaintext, plaintext_len, d_nonce, d_key);

    cudaMemcpy(h_ciphertext, d_ciphertext, num_lines * ciphertext_len * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_lines; ++i)
    {
        for (size_t j = 0; j < ciphertext_len; ++j)
        {
            output_file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(h_ciphertext[i * ciphertext_len + j]);
        }
        output_file << endl;
    }

    input_file.close();
    output_file.close();

    delete[] h_plaintext;
    delete[] h_ciphertext;
    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_nonce);
    cudaFree(d_key);

    auto end = std::chrono::high_resolution_clock::now();  // Kết thúc đo thời gian
    std::chrono::duration<double> elapsed_time = end - start;
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;  // Hiển thị thời gian chạy

    return 0;
}
