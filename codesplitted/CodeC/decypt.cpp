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
