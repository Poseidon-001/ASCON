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

__device__ void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen)
{
    // ...existing code...
}

__device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
{
    // ...existing code...
}

__device__ int ascon_compare(const uint8_t *a, const uint8_t *b, size_t len)
{
    // ...existing code...
}

__global__ void ascon_aead_decrypt_kernel(uint8_t *m, const uint8_t *t, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k, int *result)
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
    // Example data for decryption
    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
    std::vector<uint8_t> ad = {0x09, 0x0A, 0x0B, 0x0C};
    std::vector<uint8_t> ciphertext;
    std::vector<uint8_t> decrypted;
    std::vector<uint8_t> tag(16);

    // Read ciphertext from file
    std::string hex_string = read_hex_from_file("ciphertext.txt");
    hex_to_bytes(hex_string, ciphertext);
    size_t ciphertext_len = ciphertext.size();
    decrypted.resize(ciphertext_len - 16);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (auto &byte : key)
        byte = dis(gen);
    for (auto &byte : nonce)
        byte = dis(gen);

    uint8_t *d_ciphertext, *d_decrypted, *d_tag, *d_nonce, *d_key;
    int *d_result;
    cudaMalloc(&d_ciphertext, ciphertext.size());
    cudaMalloc(&d_decrypted, decrypted.size());
    cudaMalloc(&d_tag, tag.size());
    cudaMalloc(&d_nonce, nonce.size());
    cudaMalloc(&d_key, key.size());
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_ciphertext, ciphertext.data(), ciphertext.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, nonce.data(), nonce.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key.data(), key.size(), cudaMemcpyHostToDevice);

    ascon_aead_decrypt_kernel<<<1, 1>>>(d_decrypted, d_tag, d_ciphertext, ciphertext_len, ad.data(), ad.size(), d_nonce, d_key, d_result);
    cudaMemcpy(decrypted.data(), d_decrypted, decrypted.size(), cudaMemcpyDeviceToHost);
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Decryption result: " << (result == 0 ? "Success" : "Failure") << std::endl;

    cudaFree(d_ciphertext);
    cudaFree(d_decrypted);
    cudaFree(d_tag);
    cudaFree(d_nonce);
    cudaFree(d_key);
    cudaFree(d_result);

    return 0;
}
