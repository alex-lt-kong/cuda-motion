## Compilation

g++ motionDerector.cpp ./classes/deviceManager.cpp -o motionDerector -L/usr/local/lib -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core -pthread


## Environments

A few heavy components are needed for this program to be fully functional.
For all the following compilation/installation, it is almost always better to
clone the entire git repository since there could be errors in multiple cases,
trying different version is usually unavoidable...

* If there is an `FFmpeg` installed by `apt`, remove it first.
* Install NVIDIA GPU driver and make sure everything works with `nvidia-smi`.
* Install `FFmpeg` 4.4 with NVIDIA Cuda support following NVIDIA's official guide: https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/. Note that as of February 2022, `FFmpeg` 4.5 does not seem to work since it appears to be incompatible with `OpenCV`.
* Install `OpenCV`.