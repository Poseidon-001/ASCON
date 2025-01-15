#include "ascon.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <fstream>
#include <cuda_runtime.h>

// Define RATE based on the variant
#ifdef ASCON_AEAD_RATE
#define RATE ASCON_AEAD_RATE
#else
#define RATE 8
#endif

using namespace std;

// Helper functions
__device__ void ascon_permutation(ascon_state_t *s, int rounds);

// Ascon permutation function
__device__ void ascon_permutation(ascon_state_t *s, int rounds)
{
    // ...existing code...
}

// AEAD functions
__device__ void ascon_loadkey(ascon_key_t *key, const uint8_t *k)
{
    // ...existing code...
}

__device__ void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub)
{
    // ...existing code...
}

__device__ void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen)
{
    // ...existing code...
}

__device__ void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen)
{
    // ...existing code...
}

__device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
{
    // ...existing code...
}

__global__ void ascon_aead_encrypt_kernel(uint8_t *t, uint8_t *c, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    // ...existing code...
}

// Helper function to convert integer to hex string
string toHex(int val)
{
    // ...existing code...
}

// Helper function to convert hex string to byte array
void hex_to_bytes(const std::string &hex, std::vector<uint8_t> &bytes)
{
    // ...existing code...
}

// Function to read hex string from file
std::string read_hex_from_file(const std::string &filename)
{
    // ...existing code...
}

int main()
{
    // Example data for encryption
    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
    std::vector<uint8_t> ad = {0x09, 0x0A, 0x0B, 0x0C};
    std::vector<uint8_t> plaintext;
    std::vector<uint8_t> ciphertext;
    std::vector<uint8_t> tag(16);

    // Read plaintext from file
    std::string hex_string = read_hex_from_file("plaintext.txt");
    hex_to_bytes(hex_string, plaintext);
    size_t plaintext_len = plaintext.size();
    ciphertext.resize(plaintext_len + 16);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (auto &byte : key)
        byte = dis(gen);
    for (auto &byte : nonce)
        byte = dis(gen);

    uint8_t *d_plaintext, *d_ciphertext, *d_tag, *d_nonce, *d_key;
    cudaMalloc(&d_plaintext, plaintext.size());
    cudaMalloc(&d_ciphertext, ciphertext.size());
    cudaMalloc(&d_tag, tag.size());
    cudaMalloc(&d_nonce, nonce.size());
    cudaMalloc(&d_key, key.size());

    cudaMemcpy(d_plaintext, plaintext.data(), plaintext.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce.data(), nonce.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice);

    ascon_aead_encrypt_kernel<<<1, 1>>>(d_tag, d_ciphertext, d_plaintext, plaintext_len, ad.data(), ad.size(), d_nonce, d_key);
    cudaMemcpy(tag.data(), d_tag, tag.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ciphertext.data(), d_ciphertext, ciphertext.size(), cudaMemcpyDeviceToHost);

    std::ofstream output_file("ciphertext.txt");
    for (auto byte : ciphertext)
    {
        output_file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    output_file << endl;

    cudaFree(d_plaintext);
    cudaFree(d_ciphertext);
    cudaFree(d_tag);
    cudaFree(d_nonce);
    cudaFree(d_key);

    return 0;
}
