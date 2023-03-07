# Camera server

A C++17 project inspired by, similar to but simpler than
[Motion](https://github.com/Motion-Project/motion) in C.

## Dependencies

* [Crow HTTP library](https://github.com/CrowCpp/Crow) for HTTP service support
  * `Asio`, an  asynchronous mode used by Crow:  `apt install libasio-dev`
* `nlohmann-json3 (>= 3.9)`, JSON support: `apt install nlohmann-json3-dev`
* `opencv`, for frame manipulation: `apt install libopencv-dev`.
* `v4l-utils`: for manually examining and manipulating local video devices.

### FFmpeg

* FFmpeg is the back-end used by `OpenCV` to decode/encode videos.
* If you don't have an Nvidia GPU, simply issue `apt install ffmpeg` should
be enough--we will use FFmpeg's default configuration and use the CPU to do
all the heavy-lifting things. Apart from `FFmpeg` itself, a few libraries
used by it should also be installed: 
  * `libavcodec` provides implementation of a wider range of codecs.
  * `libavformat` implements streaming protocols, container formats and basic I/O access.
  * `libavutil` includes hashers, decompressors and miscellaneous utility functions.
  * `libavfilter` provides a mean to alter decoded Audio and Video through chain of filters.
  * `libavdevice` provides an abstraction to access capture and playback devices.
  * `libswresample` implements audio mixing and resampling routines.
  

* Otherwise, it is going to be much more complicated as we need to make FFmpeg work with the GPU:
  * If there is an `FFmpeg` installed by `apt`, remove it first.
  * Install an Nvidia GPU driver and make sure everything works with `nvidia-smi`.
  * Install `FFmpeg` 4.4 with Nvidia Cuda support following Nvidia's official guide:
  https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/. Note that as of February 2022,
  `FFmpeg` 4.5 does not seem to work since it appears to be incompatible with `OpenCV`.
  * There are tons of parameters to tweak while using FFmpeg with Nvidia GPUs, [this doc](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/) is a good starting point.
  * Important observation: Even with a GPU enabled, directly encoding incoming frames from cameras to
  two destination video files causes performance to drop significantly, if two resolutions are needed,
  one should consider transcoding with scaling after the first and larger video is successfully encoded.


## Explanation of Some Confusing Parameters

* `preferredWidth`,`preferredHeight`, `preferredFps`: If positive, they will be directly passed to `OpenCV`'s `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT` and `CAP_PROP_FPS`. It is subject to `OpenCV`'s discretion on how they are interpreted. Usually
they are only effective when the source is a local Linux video device.

## Useful Commands

### 1. `v4l2-ctl`

* List supported video devices: `v4l2-ctl --list-devices`
* List supported resolutions of a video device: `v4l2-ctl --list-formats-ext --device <videoUri>`
* Get pixel format from a video device: `v4l2-ctl --get-fmt-video --device <videoUri>`
* Set pixel format to `MJPG` to a video device: `v4l2-ctl --set-fmt-video=pixelformat=MJPG --device <videoUri>`
* Get all parameters `v4l2-ctl --get-parm --all --device <videoUri>`
* Set a parameter: `v4l2-ctl --set-ctrl=<parameterName>=<parameterValue>`
* Get framerate: `v4l2-ctl --get-parm --device <videoUri>`
* Set framerate: `v4l2-ctl --set-parm=30 --device <videoUri>`

### 2. `ffmpeg` and `ffprobe`

#### Ascertain the `-pix_fmt` option is `ffmpeg` command

* Seems that there isn't a credible way lol.
* Issue `ffmpeg -pix_fmts` and try the results one by one.
* Some options are more likely to be the right value than others, such as `yuv420p`, `yuyv422`, `bgr24`, `rgb24`.

#### Ascertain the FPS of a video source
```
# ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate <videoUri>
25/1
```
* Note that this value may not be accurate for remote video sources.
* A more accurate but less formal way is to simply observe the output from `ffmpeg` itself.
