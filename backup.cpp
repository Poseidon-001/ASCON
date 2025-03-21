#include "ascon.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>
#include <opencv2/opencv.hpp>

// Define RATE based on the variant
#ifdef ASCON_AEAD_RATE
#define RATE ASCON_AEAD_RATE
#else
#define RATE 8
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Helper functions
static void ascon_permutation(ascon_state_t *s, int rounds);

// Ascon permutation function
static void ascon_permutation(ascon_state_t *s, int rounds)
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
    uint64_t cblock = ((uint64_t *)lastblock)[0];
    ((uint64_t *)m)[0] = s->x[0] ^ cblock;
    s->x[0] = cblock;
}

void ascon_final(ascon_state_t *s, const ascon_key_t *k)
{
    s->x[1] ^= k->x[0];
    s->x[2] ^= k->x[1];
    ascon_permutation(s, 12);
    s->x[3] ^= k->x[0];
    s->x[4] ^= k->x[1];
}

int ascon_aead_encrypt(uint8_t *t, uint8_t *c, const uint8_t *m, uint64_t mlen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);
    ascon_encrypt(&s, c, m, mlen);
    ascon_final(&s, &key);
    std::memcpy(t, &s.x[3], 16);
    return 0;
}

int ascon_aead_decrypt(uint8_t *m, const uint8_t *t, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k)
{
    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);
    ascon_decrypt(&s, m, c, clen - 16);
    ascon_final(&s, &key);
    return std::memcmp(t, &s.x[3], 16) == 0 ? 0 : -1;
}

// Hash functions
void ascon_inithash(ascon_state_t *s)
{
    std::memset(s, 0, sizeof(ascon_state_t));
    s->x[0] = 0x00400c0000000100ULL;
    ascon_permutation(s, 12);
}

void ascon_absorb(ascon_state_t *s, const uint8_t *in, uint64_t inlen)
{
    while (inlen >= RATE)
    {
        s->x[0] ^= ((uint64_t *)in)[0];
        ascon_permutation(s, 12);
        in += RATE;
        inlen -= RATE;
    }
    uint8_t lastblock[RATE] = {0};
    std::memcpy(lastblock, in, inlen);
    lastblock[inlen] = 0x80;
    s->x[0] ^= ((uint64_t *)lastblock)[0];
}

void ascon_squeeze(ascon_state_t *s, uint8_t *out, uint64_t outlen)
{
    while (outlen >= RATE)
    {
        std::memcpy(out, &s->x[0], RATE);
        ascon_permutation(s, 12);
        out += RATE;
        outlen -= RATE;
    }
    std::memcpy(out, &s->x[0], outlen);
}

int ascon_xof(uint8_t *out, uint64_t outlen, const uint8_t *in, uint64_t inlen)
{
    ascon_state_t s;
    ascon_inithash(&s);
    ascon_absorb(&s, in, inlen);
    ascon_squeeze(&s, out, outlen);
    return 0;
}

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

// Helper function to print bytes in hex format
void print_hex(std::ofstream &file, const std::string &label, const uint8_t *data, size_t len)
{
    file << label << ": ";
    for (size_t i = 0; i < len; ++i)
    {
        file << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
    }
    file << " (" << len << " bytes)" << std::endl;
}

string toHex(int val)
{
    stringstream ss;
    ss << setfill('0') << setw(2) << hex << val;
    return ss.str();
}

int main() {
    cv::VideoCapture videoFace_data(0); // Use 0 for the default webcam

    if (!videoFace_data.isOpened()) {
        std::cerr << "Unable to open webcam" << std::endl;
        return -1;
    }

    int fps = 24;
    int frame_delay = 1000 / fps;

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_alt2.xml"))) {
        std::cerr << "Error loading face cascade" << std::endl;
        return -1;
    }

    cv::Size fixed_size(200, 200); // Set a fixed size for the ROI
    int frame_count = 0;
    int no_face_count = 0; // Counter for frames with no face detected
    const int max_no_face_frames = 30; // Maximum number of consecutive frames with no face before stopping

    std::ofstream output_file("face_detection.txt");
    if (!output_file.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        return -1;
    }

    std::vector<uint8_t> key(16);
    std::vector<uint8_t> nonce(16);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (auto &byte : key)
        byte = dis(gen);
    for (auto &byte : nonce)
        byte = dis(gen);

    // Output folder path
    std::string output_folder = "E:\\ASCON\\Flow\\FLow_ascon\\image";

    // Initialize video writers
    cv::VideoWriter dot_video_writer(output_folder + "\\dot_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, fixed_size);
    cv::VideoWriter face_video_writer(output_folder + "\\face_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size((int)videoFace_data.get(cv::CAP_PROP_FRAME_WIDTH), (int)videoFace_data.get(cv::CAP_PROP_FRAME_HEIGHT)));

    if (!dot_video_writer.isOpened() || !face_video_writer.isOpened()) {
        std::cerr << "Error opening video writers" << std::endl;
        return -1;
    }

    auto total_start = std::chrono::steady_clock::now();

    while (videoFace_data.isOpened()) {
        auto frame_start = std::chrono::steady_clock::now();

        cv::Mat frame;
        videoFace_data >> frame;
        if (frame.empty()) {
            break;
        }

        // Write the original frame to the face video
        face_video_writer.write(frame);

        // Define a fixed ROI in the center of the frame
        int x = (frame.cols - fixed_size.width) / 2;
        int y = (frame.rows - fixed_size.height) / 2;
        cv::Rect roi(x, y, fixed_size.width, fixed_size.height);

        // Crop the frame to the fixed ROI
        cv::Mat face_crop = frame(roi);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(face_crop, faces, 1.1, 5, 0, cv::Size(30, 30));

        if (faces.empty()) {
            no_face_count++;
            if (no_face_count >= max_no_face_frames) {
                std::cout << "No face detected for " << max_no_face_frames << " consecutive frames. Stopping video." << std::endl;
                break;
            }
        } else {
            no_face_count = 0; // Reset the counter if a face is detected

            // Convert face_crop to a byte array
            std::vector<uint8_t> plaintext(face_crop.total() * face_crop.elemSize());
            std::memcpy(plaintext.data(), face_crop.data, plaintext.size());

            size_t plaintext_len = plaintext.size();
            std::vector<uint8_t> ciphertext(plaintext_len + 16);
            std::vector<uint8_t> tag(16);
            std::vector<uint8_t> decrypted(plaintext_len);

            ascon_aead_encrypt(tag.data(), ciphertext.data(), plaintext.data(), plaintext_len, nullptr, 0, nonce.data(), key.data());

            int result = ascon_aead_decrypt(decrypted.data(), tag.data(), ciphertext.data(), plaintext_len + 16, nullptr, 0, nonce.data(), key.data());

            output_file << "Frame " << frame_count << std::endl;
            // print_hex(output_file, "key", key.data(), key.size());
            // print_hex(output_file, "nonce", nonce.data(), nonce.size());
            // print_hex(output_file, "plaintext", plaintext.data(), plaintext_len);
            // print_hex(output_file, "ciphertext", ciphertext.data(), plaintext_len);
            // print_hex(output_file, "tag", tag.data(), tag.size());
            // print_hex(output_file, "received", decrypted.data(), plaintext_len);

            cv::Mat color_dot_image(fixed_size.height, fixed_size.width, CV_8UC3);
            for (int i = 0; i < fixed_size.height; ++i) {
                for (int j = 0; j < fixed_size.width; ++j) {
                    int index = (i * fixed_size.width + j) * 3;
                    if (index + 2 < ciphertext.size()) {
                        color_dot_image.at<cv::Vec3b>(i, j) = cv::Vec3b(ciphertext[index], ciphertext[index + 1], ciphertext[index + 2]);
                    } else {
                        color_dot_image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0); // Padding with black if out of bounds
                    }
                }
            }
            dot_video_writer.write(color_dot_image);

            // Display the dot video
            cv::imshow("Color Dot Image", color_dot_image);

            frame_count++;

            if (cv::waitKey(frame_delay) & 0xFF == 'q') {
                break;
            }
        }

        auto frame_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> frame_elapsed = frame_end - frame_start;
        output_file << "Total processing time: " << frame_elapsed.count() << " seconds" << std::endl;
    }
    videoFace_data.release();
    dot_video_writer.release();
    face_video_writer.release();
    cv::destroyAllWindows();

    output_file.close();
    std::cout << "Hex strings saved to face_detection.txt" << std::endl;

    return 0;
}
