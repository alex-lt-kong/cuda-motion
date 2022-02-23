#include <iostream>                                                                                                    
#include <vector>                        
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"              

using namespace std;
using namespace cv;                                                          

int main() {   
    
    bool result;
    VideoCapture cap = VideoCapture();

    result = cap.open("");
    if (result == false) { cout << "Failed to open, returned." << endl; return 1;}
    
    FILE *output;
    output = popen ("/usr/bin/ffmpeg -y -f rawvideo -pixel_format bgr24 -video_size 640x480 -framerate 10 -i pipe:0 -vcodec h264 /tmp/test.mp4", "w");                                           
    //output = popen ("/usr/local/bin/ffmpeg -y -f rawvideo  -pixel_format bgr24 -hwaccel cuda -hwaccel_output_format cuda -video_size 1920x1080 -framerate 15 -i pipe:0 -vcodec h264_nvenc /tmp/test.mp4", "w")
    Mat frame;                                                               
    std::ios::sync_with_stdio(false);                                        
    int count = 0;
    while (cap.read(frame)) {                                                
        for (size_t i = 0; i < frame.dataend - frame.datastart; i++) 
     //      std::cout << frame.data[i];
           fwrite(&frame.data[i], sizeof(frame.data[i]), 1, output);
        count ++;
        if (count > 100) { break; }                                   
    }  
    pclose(output);                                                                      
    return 0;                                                                
} 