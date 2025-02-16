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
#include <cuda_runtime.h>

// Define RATE based on the variant
#ifdef ASCON_AEAD_RATE
#define RATE ASCON_AEAD_RATE
#else
#define RATE 8
#endif

using namespace cv;
using namespace std;

// Helper functions
__device__ uint64_t warp_shuffle(uint64_t val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__constant__ uint8_t RC[12] = {0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4};

__device__ void ascon_permutation(ascon_state_t *s, int rounds) {
    uint64_t x0 = s->x[0];
    uint64_t x1 = s->x[1];
    uint64_t x2 = s->x[2];
    uint64_t x3 = s->x[3];
    uint64_t x4 = s->x[4];

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
__host__ __device__ void ascon_loadkey(ascon_key_t *key, const uint8_t *k)
{
    memcpy(key->b, k, CRYPTO_KEYBYTES);
}

__host__ __device__ void ascon_initaead(ascon_state_t *s, const ascon_key_t *key, const uint8_t *npub)
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

__host__ __device__ void ascon_adata(ascon_state_t *s, const uint8_t *ad, uint64_t adlen)
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

__host__ __device__ void ascon_encrypt(ascon_state_t *s, uint8_t *c, const uint8_t *m, uint64_t mlen)
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

__host__ __device__ void ascon_decrypt(ascon_state_t *s, uint8_t *m, const uint8_t *c, uint64_t clen)
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

__host__ __device__ void ascon_final(ascon_state_t *s, const ascon_key_t *k)
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = mlen / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = (idx == gridDim.x * blockDim.x - 1) ? mlen : start + chunk_size;

    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);

    uint64_t x0 = s.x[0];
    uint64_t x1 = s.x[1];
    uint64_t x2 = s.x[2];
    uint64_t x3 = s.x[3];
    uint64_t x4 = s.x[4];

    for (int i = start; i < end; i += ASCON_AEAD_RATE) {
        x0 ^= ((uint64_t *)(m + i))[0];
        ((uint64_t *)(c + i))[0] = x0;
        ascon_permutation(&s, 6);
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

    ascon_final(&s, &key);
    if (idx == 0) {
        memcpy(t, &s.x[3], 16);
    }
}

__global__ void ascon_aead_decrypt_kernel(uint8_t *m, const uint8_t *t, const uint8_t *c, uint64_t clen, const uint8_t *ad, uint64_t adlen, const uint8_t *npub, const uint8_t *k, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = clen / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = (idx == gridDim.x * blockDim.x - 1) ? clen : start + chunk_size;

    ascon_state_t s;
    ascon_key_t key;
    ascon_loadkey(&key, k);
    ascon_initaead(&s, &key, npub);
    ascon_adata(&s, ad, adlen);

    uint64_t x0 = s.x[0];
    uint64_t x1 = s.x[1];
    uint64_t x2 = s.x[2];
    uint64_t x3 = s.x[3];
    uint64_t x4 = s.x[4];

    for (int i = start; i < end; i += ASCON_AEAD_RATE) {
        uint64_t cblock = ((uint64_t *)(c + i))[0];
        ((uint64_t *)(m + i))[0] = x0 ^ cblock;
        x0 = cblock;
        ascon_permutation(&s, 6);
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

    ascon_final(&s, &key);
    if (idx == 0) {
        *result = ascon_compare(t, (uint8_t *)&s.x[3], 16);
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

int main()
{
    // Change to read from a video file
    std::string video_path = "path_to_your_video_file.mp4";
    cv::VideoCapture videoFace_data(video_path);

    if (!videoFace_data.isOpened()) {
        std::cerr << "Unable to open video file" << std::endl;
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

            // Measure encryption time
            auto start_encrypt = std::chrono::high_resolution_clock::now();
            ascon_aead_encrypt_kernel<<<1, 1>>>(d_tag, d_ciphertext, d_plaintext, plaintext_len, nullptr, 0, d_nonce, d_key);
            cudaDeviceSynchronize();
            auto end_encrypt = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
            std::cout << "Encryption time: " << elapsed_encrypt.count() << " seconds" << std::endl;

            cudaMemcpy(tag.data(), d_tag, tag.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(ciphertext.data(), d_ciphertext, ciphertext.size(), cudaMemcpyDeviceToHost);

            // Measure decryption time
            auto start_decrypt = std::chrono::high_resolution_clock::now();
            ascon_aead_decrypt_kernel<<<1, 1>>>(d_plaintext, d_tag, d_ciphertext, plaintext_len + 16, nullptr, 0, d_nonce, d_key, d_result);
            cudaDeviceSynchronize();
            auto end_decrypt = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_decrypt = end_decrypt - start_decrypt;
            std::cout << "Decryption time: " << elapsed_decrypt.count() << " seconds" << std::endl;

            cudaMemcpy(decrypted.data(), d_plaintext, decrypted.size(), cudaMemcpyDeviceToHost);
            int result;
            cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

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

            cudaFree(d_plaintext);
            cudaFree(d_ciphertext);
            cudaFree(d_tag);
            cudaFree(d_nonce);
            cudaFree(d_key);
            cudaFree(d_result);
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
