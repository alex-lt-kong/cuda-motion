## Compilation

```
g++ ./src/main.cpp ./src/classes/deviceManager.cpp ./src/classes/motionDetector.cpp ./src/classes/logger.cpp -o motionDetector  -L/usr/local/lib -I/usr/local/include/opencv4 -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core -pthread
```

## Environments

A few heavy components are needed for this program to be fully functional:

* `OpenCV`: Used to manipulate frames.
* `ffmpeg`: To leverage NVidia's hardware, `ffmpeg` is always used to encode video.

For all the following compilation/installation, it is almost always better to
clone the entire git repository since there could be errors in multiple cases,
trying different version is usually unavoidable...

* If there is an `FFmpeg` installed by `apt`, remove it first.
* Install NVIDIA GPU driver and make sure everything works with `nvidia-smi`.
* Install `FFmpeg` 4.4 with NVIDIA Cuda support following NVIDIA's official guide: https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/. Note that as of February 2022, `FFmpeg` 4.5 does not seem to work since it appears to be incompatible with `OpenCV`.
* Install `OpenCV`.

```
apt install nlohmann-json3-dev libspdlog-dev
```

# Useful Commands

## List supported resolutions of video source:

* `v4l2-ctl -d [videoUri] --list-formats-ext`

## Ascertain the `-pix_fmt` option is `ffmpeg` command

* Seems that there isn't a credible way lol.
* Issue `ffmpeg -pix_fmts` and try the results one by one.
* Some options are more likely to be the right value than others, such as `yuv420p`, `yuyv422`, `bgr24`, `rgb24`.

## Ascertain the FPS of a video source
```
# ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate [videoUri]
25/1
```
Note that this data may or may not be accurate.