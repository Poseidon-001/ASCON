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

// CUDA kernel for Ascon encryption
__global__ void ascon_encrypt_kernel(uint8_t *d_ciphertext, const uint8_t *d_plaintext, size_t plaintext_len, const uint8_t *d_nonce, const uint8_t *d_key)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < plaintext_len)
    {
        ascon_state_t s;
        ascon_key_t key;
        ascon_loadkey(&key, d_key);
        ascon_initaead(&s, &key, d_nonce);
        ascon_adata(&s, nullptr, 0);
        ascon_encrypt(&s, d_ciphertext + idx * (plaintext_len + 16), d_plaintext + idx * plaintext_len, plaintext_len);
        ascon_final(&s, &key);
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
