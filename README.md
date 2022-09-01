# Motion Detector

A C++ project inspired by, similar to but simpler than [Motion](https://github.com/Motion-Project/motion) in C.


## Environments

A few heavy components are needed for this program to be fully functional:

* `OpenCV`: Used to manipulate frames.
* `ffmpeg`: To leverage Nvidia's hardware, `ffmpeg` is always used to encode video.
* For all the following compilation/installation, it is almost always better to
clone the entire git repository since there could be errors in multiple cases,
trying different version is usually unavoidable...

### OpenCV

* Follow [this link](https://github.com/alex-lt-kong/q-rtsp-viewer)

### FFmpeg

* If you don't have a GPU, simply issue `apt install ffmpeg` should be enough.

Otherwise...it is going to be much more complicated:
  * If there is an `FFmpeg` installed by `apt`, remove it first.
  * Install Nvidia GPU driver and make sure everything works with `nvidia-smi`.
  * Install `FFmpeg` 4.4 with Nvidia Cuda support following Nvidia's official guide: https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/. Note that as of February 2022, `FFmpeg` 4.5 does not seem to work since it appears to be incompatible with `OpenCV`.

* There are tons of parameters to tweak while using FFmpeg with Nvidia GPUs, [this doc](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/) is a good starting point.

* Directly encoding incoming frames from cameras to two destination video files causes performace to drop significantly,
if two resolutions are needed, one should consider transcoding with scaling after the first and larger video is
successfully encoded.

### Misc

* Install `nlohmann-json3` for JSON support: `apt install nlohmann-json3-dev`

### Environment Variables

* Sometimes you may be able to compile the project but running `motionDetector` gives `./motionDetector: error while loading shared libraries: libopencv_imgcodecs.so.405: cannot open shared object file: No such file or directory`
* One solution is to add `/usr/local/lib/` to `LD_LIBRARY_PATH`: `export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH`


## Explanation to Some Confusing Parameters

* `preferredWidth`,`preferredHeight`, `preferredFps`: If positive, they will be directly passed to `OpenCV`'s `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT` and `CAP_PROP_FPS`. It is subject to `OpenCV`'s discretion on how they are interpreted. Usually
they are only effective when the source is a local Linux video device.
* `FpsUpperCap`: If the motionDetector gets frames faster than this value, it simply discard the frame and read the next 
one, skipping all further processing. It is useful to limit the CPU usage of the program when FPS from a video device
cannot be controlled by `preferredFps`.

## Useful Commands

### 1. `v4l2-ctl`

* List supported video devices: `v4l2-ctl --list-devices`
* List supported resolutions of a video device: `v4l2-ctl --list-formats-ext --device <videoUri>`
* Get pixel format from a video device: `v4l2-ctl --get-fmt-video --device <videoUri>`
* Set pixel format to `MJPG` to a video device: `v4l2-ctl --set-fmt-video=pixelformat=MJPG --device <videoUri>`
* Get all parameters `v4l2-ctl --get-parm --all --device <videoUri>`
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
