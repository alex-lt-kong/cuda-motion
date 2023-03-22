# Camera server

A C++17 project inspired by, similar to but simpler than
[Motion](https://github.com/Motion-Project/motion).

There are two main functions of this program:

1. Detect image change (i.e., "motion") from a video feed and record videos
during the periods when a motion event happens;
1. Provide various inter-process communication methods for downstream
programs to consume live images from different types of cameras easily
so that downstream programs don't have to implement their own version of video
feed handling repetitively.

## Dependencies

* [Crow HTTP library](https://github.com/CrowCpp/Crow) for HTTP service support
  * `Asio`, an  asynchronous mode used by Crow:  `apt install libasio-dev`
  * `OpenSSL`, for SSL support: `apt-get install libssl-dev`
* `nlohmann-json3 (>= 3.9)`, JSON support: `apt install nlohmann-json3-dev`
* `opencv`, for frame manipulation: `apt install libopencv-dev`
* `spdlog` for logging: `apt install libspdlog-dev`
* `v4l-utils`: for manually examining and manipulating local video devices.
* `FFmpeg`
  * FFmpeg is the back-end used by `OpenCV` to decode/encode videos.
  * If you don't have an Nvidia GPU, simply issue `apt install ffmpeg` should
  be enough--we will use FFmpeg's default configuration and use the CPU to do
  all the heavy-lifting things.

  * Otherwise, it is going to be much more complicated as we need to make
  FFmpeg work with the GPU:
    * If there is an `FFmpeg` installed by `apt`, remove it first.
    * Install an Nvidia GPU driver and make sure everything works with
    `nvidia-smi`.
    * Install `FFmpeg` 4.4 with Nvidia Cuda support following Nvidia's
    [official guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/).
    Note that as of February 2022, `FFmpeg` 4.5 does not seem to work since
    it appears to be incompatible with `OpenCV`.
    * There are tons of parameters to tweak while using FFmpeg with Nvidia GPUs,
    [this doc](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/) is a good starting point.
    * Important observation: Even with a GPU enabled, directly encoding
    incoming frames from cameras to two destination video files causes
    performance to drop significantly, if two resolutions are needed, one
    should consider transcoding with scaling after the first and larger video
    is successfully encoded.

## Build and deployment


* 
```bash
mkdir ./build
cmake ../
make -j2
```
* Copy `./configs/camera-server.jsonc` to `~/.configs/ak-studio`.
* The program is tested on Debian and should work on other distributions
or POSIX-compliant OSes. However, given that it uses quite a few POSIX APIs,
it is unlikely that it could run on Windows without significant porting effort.

## Quality assurance

* Instead of `cmake ../`, run `cmake .. -DBUILD_ASAN=ON` /
`cmake .. -DBUILD_UBSAN=ON ` to test memory/undefiend behavior error with
AddressSanitizer / UndefinedBehaviorSanitizer.
* The repo is also tested with `Valgrind` from time to time:
`valgrind --leak-check=yes --log-file=valgrind.rpt ./build/cs`.
