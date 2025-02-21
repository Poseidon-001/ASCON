#include "ascon.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>

// Define RATE based on the variant
#ifdef ASCON_AEAD_RATE
#define RATE ASCON_AEAD_RATE
#else
#define RATE 8
#endif

using namespace std;

// Helper functions
void ascon_permutation(ascon_state_t *s, int rounds) {
    static const uint8_t RC[12] = {0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4};
    uint64_t x0 = s->x[0];
    uint64_t x1 = s->x[1];
    uint64_t x2 = s->x[2];
    uint64_t x3 = s->x[3];
    uint64_t x4 = s->x[4];

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
void ascon_loadkey(ascon_key_t *key, const uint8_t *k) {
    memcpy(key->b, k, CRYPTO_KEYBYTES);
}

void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub) {
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

void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen) {
    while (adlen >= ASCON_AEAD_RATE) {
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

void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen) {
    while (mlen >= ASCON_AEAD_RATE) {
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

void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen) {
    while (clen >= ASCON_AEAD_RATE) {
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

void ascon_final(ascon_state_t *s, const ascon_key_t *k) {
    s->x[1] ^= k->x[0];
    s->x[2] ^= k->x[1];
    ascon_permutation(s, 12);
    s->x[3] ^= k->x[0];
    s->x[4] ^= k->x[1];
}

int ascon_compare(const uint8_t *a, const uint8_t *b, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (a[i] != b[i]) {
            return -1;
        }
    }
    return 0;
}

// Helper function to convert hex string to byte array
void hex_to_bytes(const std::string &hex, std::vector<uint8_t> &bytes) {
    bytes.resize(hex.length() / 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        std::stringstream ss;
        ss << std::hex << hex.substr(2 * i, 2);
        int byte;
        ss >> byte;
        bytes[i] = static_cast<uint8_t>(byte);
    }
}

// Function to read hex string from file
std::string read_hex_from_file(const std::string &filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    // Read hex input from file
    std::string hex_input = read_hex_from_file("input_hex.txt");
    std::vector<uint8_t> plaintext;
    hex_to_bytes(hex_input, plaintext);

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

    // Measure encryption time
    auto start_encrypt = std::chrono::high_resolution_clock::now();
    ascon_state_t s;
    ascon_key_t key_struct;
    ascon_loadkey(&key_struct, key.data());
    ascon_initaead(&s, &key_struct, nonce.data());
    ascon_adata(&s, nullptr, 0);
    ascon_encrypt(&s, ciphertext.data(), plaintext.data(), plaintext_len);
    ascon_final(&s, &key_struct);
    memcpy(tag.data(), &s.x[3], 16);
    auto end_encrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
    std::cout << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;

    // Measure decryption time
    auto start_decrypt = std::chrono::high_resolution_clock::now();
    ascon_initaead(&s, &key_struct, nonce.data());
    ascon_adata(&s, nullptr, 0);
    ascon_decrypt(&s, decrypted.data(), ciphertext.data(), plaintext_len);
    ascon_final(&s, &key_struct);
    int result = ascon_compare(tag.data(), (uint8_t *)&s.x[3], 16);
    auto end_decrypt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_decrypt = end_decrypt - start_decrypt;
    std::cout << "Decryption time: " << elapsed_decrypt.count() << " seconds" << std::endl;

    std::ofstream output_file("output.txt");
    if (!output_file.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        return -1;
    }

    output_file << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;
    output_file << "Decryption time: " << elapsed_decrypt.count() << " seconds" << std::endl;
    output_file << "Decryption result: " << (result == 0 ? "Success" : "Failure") << std::endl;

    output_file.close();
    std::cout << "Results saved to output.txt" << std::endl;

    return 0;
}
