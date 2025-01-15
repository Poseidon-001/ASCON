#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Helper function to convert integer to hex string
string toHex(int val)
{
    stringstream ss;
    ss << setfill('0') << setw(2) << hex << val;
    return ss.str();
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

    ofstream output_file("frames.txt");
    if (!output_file.is_open())
    {
        cerr << "Unable to open file for writing" << endl;
        return -1;
    }

    while (video.isOpened())
    {
        video >> frame;
        if (frame.empty())
        {
            break;
        }

        for (int y = 0; y < frame.rows; ++y)
        {
            for (int x = 0; x < frame.cols; ++x)
            {
                Vec3b pixel = frame.at<Vec3b>(y, x);
                output_file << toHex(pixel[2]) << toHex(pixel[1]) << toHex(pixel[0]);
            }
        }
        output_file << endl;

        imshow("Video", frame);
        if (waitKey(frame_delay) & 0xFF == 'q')
        {
            break;
        }
    }

    video.release();
    destroyAllWindows();
    output_file.close();

    return 0;
}
