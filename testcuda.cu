#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

// Helper function to convert integer to hex string
string toHex(int val)
{
    stringstream ss;
    ss << setfill('0') << setw(2) << hex << val;
    return ss.str();
}

// Kernel function to convert image to hex string
__global__ void imageToHexKernel(uchar3 *d_image, int width, int height, char *d_hexString)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        uchar3 pixel = d_image[idx];
        int hexIdx = idx * 6;
        sprintf(&d_hexString[hexIdx], "%02x%02x%02x", pixel.x, pixel.y, pixel.z);
    }
}

int main()
{
    string path = "./image/video_1.mp4";
    VideoCapture video(path);

    if (!video.isOpened())
    {
        cerr << "Unable to open video" << endl;
        return -1;
    }

    int fps = 24;
    int frame_delay = 1000 / fps;

    Mat frame;
    video >> frame;
    if (frame.empty())
    {
        cerr << "Empty frame" << endl;
        return -1;
    }

    int width = frame.cols;
    int height = frame.rows;
    size_t imageSize = width * height * sizeof(uchar3);
    size_t hexStringSize = width * height * 6 * sizeof(char);

    uchar3 *d_image;
    char *d_hexString;
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_hexString, hexStringSize);

    while (video.isOpened())
    {
        video >> frame;
        if (frame.empty())
        {
            break;
        }

        cudaMemcpy(d_image, frame.ptr<uchar3>(), imageSize, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        imageToHexKernel<<<gridSize, blockSize>>>(d_image, width, height, d_hexString);

        char *h_hexString = new char[hexStringSize];
        cudaMemcpy(h_hexString, d_hexString, hexStringSize, cudaMemcpyDeviceToHost);

        cout << "Frame Hex String: " << endl;
        for (int i = 0; i < hexStringSize; i += 6)
        {
            cout << string(&h_hexString[i], 6) << " ";
            if ((i / 6 + 1) % width == 0)
            {
                cout << endl;
            }
        }

        delete[] h_hexString;

        imshow("Video", frame);
        if (waitKey(frame_delay) & 0xFF == 'q')
        {
            break;
        }
    }

    cudaFree(d_image);
    cudaFree(d_hexString);
    video.release();
    destroyAllWindows();

    return 0;
}
