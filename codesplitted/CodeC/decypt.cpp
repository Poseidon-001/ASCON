#include "ascon.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
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

// Ascon functions (same as in the original file)
void ascon_permutation(ascon_state_t *s, int rounds)
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

void ascon_loadkey(ascon_key_t *key, const uint8_t *k)
{
    std::memcpy(key->b, k, CRYPTO_KEYBYTES);
}

void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub)
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

void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen)
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

void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen)
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

void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen)
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
    std::memcpy(lastblock, c, clen);
    lastblock[clen] = 0x80;
    uint64_t cblock = s->x[0] ^ ((uint64_t *)lastblock)[0];
    std::memcpy(m, &cblock, clen);
    s->x[0] = cblock;
}

void ascon_final(ascon_state_t *s, const ascon_key_t *key, uint8_t *tag)
{
    s->x[1] ^= key->x[0];
    s->x[2] ^= key->x[1];
    ascon_permutation(s, 12);
    s->x[3] ^= key->x[0];
    s->x[4] ^= key->x[1];
    std::memcpy(tag, &s->x[3], CRYPTO_ABYTES);
}

int ascon_aead_encrypt(uint8_t *c, uint8_t *tag, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);
    ascon_encrypt(&s, c, m, mlen);
    ascon_final(&s, &key, tag);
    return 0;
}

int ascon_aead_decrypt(uint8_t *m, uint8_t *tag, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);
    ascon_decrypt(&s, m, c, clen - CRYPTO_ABYTES);
    ascon_final(&s, &key, tag);
    return std::memcmp(tag, c + clen - CRYPTO_ABYTES, CRYPTO_ABYTES) == 0 ? 0 : -1;
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

    std::ifstream input_file("ciphertext.txt");
    if (!input_file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        return -1;
    }

    int width = 640; // Set the width of the frame
    int height = 480; // Set the height of the frame
    int fps = 24;
    int frame_delay = 1000 / fps;

    cv::VideoWriter video_writer("decrypted_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if (!video_writer.isOpened())
    {
        std::cerr << "Error opening video writer" << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(input_file, line))
    {
        std::vector<uint8_t> ciphertext;
        hex_to_bytes(line, ciphertext);
        size_t ciphertext_len = ciphertext.size();
        std::vector<uint8_t> decrypted(ciphertext_len - 16);
        std::vector<uint8_t> tag(16);

        int result = ascon_aead_decrypt(decrypted.data(), tag.data(), ciphertext.data(), ciphertext_len, nullptr, 0, nonce.data(), key.data());

        if (result != 0)
        {
            std::cerr << "Decryption failed" << std::endl;
            return -1;
        }

        cv::Mat frame(height, width, CV_8UC3, decrypted.data());
        video_writer.write(frame);

        imshow("Decrypted Video", frame);
        if (waitKey(frame_delay) & 0xFF == 'q')
        {
            break;
        }
    }

    input_file.close();
    video_writer.release();
    destroyAllWindows();

    return 0;
}
