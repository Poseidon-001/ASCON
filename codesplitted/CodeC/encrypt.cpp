#include "ascon.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <fstream>

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

// Ascon functions (same as in the original file)
// ...existing code...

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

    std::ifstream input_file("frames.txt");
    std::ofstream output_file("ciphertext.txt");
    if (!input_file.is_open() || !output_file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(input_file, line))
    {
        std::vector<uint8_t> plaintext;
        hex_to_bytes(line, plaintext);
        size_t plaintext_len = plaintext.size();
        std::vector<uint8_t> ciphertext(plaintext_len + 16);
        std::vector<uint8_t> tag(16);

        ascon_aead_encrypt(tag.data(), ciphertext.data(), plaintext.data(), plaintext_len, nullptr, 0, nonce.data(), key.data());

        for (auto byte : ciphertext)
        {
            output_file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
        }
        output_file << endl;
    }

    input_file.close();
    output_file.close();

    return 0;
}
