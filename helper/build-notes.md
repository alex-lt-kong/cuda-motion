# Build FFmpeg and OpenCV for Camera Server

## CPU route

* You follow this section if you use Camera Server with CPU only.

* Install FFmpeg: `apt install FFmpeg`

* Build OpenCV: following the same notes as [Build OpenCV](#iii-build-opencv)
in the Nvidia GPU route section.

## Nvidia GPU route

* You follow this section if you would like to use Camera Server with Nvidia GPU.

* This note is prepared using Ubuntu 22.04 with one Nvidia GPU only.

* OpenCV can work with GPUs in two ways (a good summary
[here](https://forum.opencv.org/t/trouble-using-nvdia-hardware-decoder-when-streaming-from-camera/7908/11)).
  * The first way is to specifically build OpenCV that enables
    `cv::cudacodec::VideoReader()`. This is not tested as Camera Server
    is designed to work in a wide variety of scenarios, where an Nvidia GPU
    could be available or unavailable. Using `cv::cudacodec::VideoReader()`
    means we need to align the entire program with Nvidia's ecosystem, which
    is not something we want.
  * The second is to build FFmpeg that incorporates
    [Nvidia Video Codec SDK](https://developer.nvidia.com/video-codec-sdk) and
    build OpenCV on top of FFmpeg, allowing OpenCV to leverage the power of
    GPU almost transparently.
  * Camera Server takes the second approach.

## I. Prepare Nvidia GPU

1. Make sure you have a compatible GPU of course.
    * Nvidia provides a support matrix
  [here](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)

2. Install Nvidia's GPU driver
    [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#package-manager)
nvidia-utils
3. Package `nvidia-utils-<version>` should be automatically installed by the
above steps. If so, try `nvidia-smi` to see the status of your GPU.
    * `nvtop` is a usefull third-party tool that demonstrate more GPU details
    in a manner similar to `htop`: `apt install nvtop`.

## II. Build FFmpeg

1. While OpenCV supports multiple backends, it seems that FFmpeg enjoys
better support from Nvidia, making it a good option to work with Nvidia's GPU.

1. If there is an `FFmpeg` installed by `apt`, remove it first.
    * Ubuntu does allow multiple versions to co-exist, the issue is that
    in compilation/linking, it make not be easy to configure different build
    systems (CMake/Ninja/handwritten ones) to use exactly the version of
    FFmpeg we want. Therefore, it is simpler if we can remove other
    unused FFmpeg.Z

1. Build `FFmpeg` with Nvidia GPU support **based on**
[this document](https://docs.nvidia.com/video-technologies/video-codec-sdk/pdf/Using_FFmpeg_with_NVIDIA_GPU_Hardware_Acceleration.pdf).

    * We may need to combine options gleaned from different sources to
    construct the final `./configure` command.

1. One working version of `./configure` is: `./configure --enable-pic --enable-shared --enable-nonfree --enable-cuda-sdk --enable-cuda-llvm --enable-ffnvcodec --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-cflags="-fPIC" --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags="-gencode arch=compute_52,code=sm_52 -O2"`

    * `--enable-cuda-llvm --enable-ffnvcodec`: enable Nvidia Video Codec SDK
  ([source](https://trac.ffmpeg.org/wiki/HWAccelIntro))
    * `--nvccflags="-gencode arch=compute_52,code=sm_52 -O2"`: Used to solve
  `ERROR: failed checking for nvcc.` ([source](https://docs.nvidia.com/video-technologies/video-codec-sdk/pdf/Using_FFmpeg_with_NVIDIA_GPU_Hardware_Acceleration.pdf))
        * In a future version, we may try to omit this to see if the issue
        is gone before adding it.
    * `--enable-pic --enable-shared --extra-cflags="-fPIC"`, used to solve
    the issue during OpenCV build in a later stage: "/usr/bin/ld:
    /usr/local/lib/libavcodec.a(vc1dsp_mmx.o):
    relocation R_X86_64_PC32 against symbol `ff_pw_9' can not be used when
    making a shared object; recompile with -fPIC"
    ([source1](https://www.twblogs.net/a/5ef71a3c209c567d16133dae),
    [source2](https://askubuntu.com/questions/1292968/error-when-installing-opencv-any-version-on-ubuntu-18-04))

1. If build is successful, execute `FFmpeg` and check if it works.
    * It should show something like:

    ```
    ffmpeg version n4.4.3-48-gc3ad886251 Copyright (c) 2000-2022 the FFmpeg developers
    built with gcc 11 (Ubuntu 11.3.0-1ubuntu1~22.04)
    configuration: --enable-pic --enable-shared --enable-nonfree --enable-cuda-sdk --enable-cuda-llvm --enable-ffnvcodec --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-cflags=-fPIC --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags='-gencode arch=compute_52,code=sm_52 -O2'
    libavutil      56. 70.100 / 56. 70.100
    libavcodec     58.134.100 / 58.134.100
    ...
    ```

## III. Build OpenCV

1. The first perennial problem when building OpenCV on top of a custom FFmpeg
build is compatibility--each OpenCV version depends on one or few snapshots of
FFmpeg versions, if the versions of OpenCV and FFmpeg mismatch, various
compilation errors would occur.
    * After multiple attempts, this note is prepared with FFmpeg `n4.4.3`
    and OpenCV `4.6.0`.

1. Issue `cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX="$HOME/.local"  ..`
to install the built OpenCV to the user's home directory.
    * Verify the result of `cmake` carefully before proceeding to `make`,
    make sure that `cmake` shows the expected version of `FFmpeg`.

1. If OpenCV fails to compile due to a compatibility issue with FFmpeg, we need
to go back to `II` and build FFmpeg again.
    * Before issuing `git checkout <branch/tag>`, one needs to issue
    `sudo make uninstall` and `sudo make distclean` to make sure the previous
    FFmpeg build is cleared; otherwise multiple FFmpeg versions could co-exist,
    causing confusion and strange errors.

## IV. Test

1. Enable OpenCV's debug logging, which could reveal useful information:

    ```
    export OPENCV_VIDEOIO_DEBUG=1
    export OPENCV_FFMPEG_DEBUG=1
    export OPENCV_LOG_LEVEL=DEBUG
    ```

1. Build Camera Server.

1. Enable hardware decoding for `cv::VideoCapture` with environment variable
`OPENCV_FFMPEG_CAPTURE_OPTIONS`. This envvar uses a special key/value pair
format `key1;val1|key2;val2`. To use hardware decoder, we want to set it
to something like:
    * `"hwaccel;cuvid|video_codec;mjpeg_cuvid"`--if we know the video feed is in
    MJPG format.
    * `"hwaccel;cuvid|video_codec;h264_cuvid"`--if we know the video feed is in
    h264 format.
    * List the supported codec by `ffmpeg -codecs | grep nv`.
    * It appears that all video feeds of one Camera Server must use the
    same hardware decoder, if two different video feeds use two different
    video format, we need to run two Camera Server instances.

1. Enable hardware decoding for `cv::VideoWriter` with environment variable
`OPENCV_FFMPEG_WRITER_OPTIONS`. This envvar uses a special key/value pair
format `key1;val1|key2;val2`. To use hardware encoder, we want to set it
to `"hw_encoders_any;cuda"`
    * It appears that OpenCV does not allow us to pick a specific encoder.

1. Run `./cs` then `nvidia-smi` and `nvtop`, we should see `./cs` is using GPU
now.
    * `nvidia-smi dmon -i 0 -s tu` offers some extra insights into the internals
    of the GPU in real time.

1. To be even more certain that GPU is in fact being leveraged, we can check
log of `./cs` and should see something like:
    * For video decoding (i.e., `cv::VideoCapture`), you should see log like:

    ```
    [DEBUG:2@1.037] global ./opencv/modules/videoio/src/cap_ffmpeg_impl.hpp (1229) open FFMPEG: Using video_codec='mjpeg_cuvid'
    [OPENCV:FFMPEG:40] CUVID capabilities for mjpeg_cuvid:
    [OPENCV:FFMPEG:40] 8 bit: supported: 1, min_width: 64, max_width: 32768, min_height: 64, max_height: 16384
    [OPENCV:FFMPEG:40] 10 bit: supported: 0, min_width: 0, max_width: 0, min_height: 0, max_height: 0
    [OPENCV:FFMPEG:40] 12 bit: supported: 0, min_width: 0, max_width: 0, min_height: 0, max_height: 0
    ```

    * For video encoding (i.e., `cv::VideoWriter`), you should see log like:

    ```
    [DEBUG:0@12.887] global ./opencv/modules/videoio/src/cap_ffmpeg_hw.hpp (933) HWAccelIterator FFMPEG: allowed acceleration types (any): 'cuda,'
    [DEBUG:0@12.887] global ./opencv/modules/videoio/src/cap_ffmpeg_hw.hpp (951) HWAccelIterator FFMPEG: disabled codecs: 'mjpeg_vaapi,mjpeg_qsv,vp8_vaapi'
    [ INFO:0@12.984] global ./opencv/modules/videoio/src/cap_ffmpeg_hw.hpp (278) hw_check_device FFMPEG: Using cuda video acceleration
    [ INFO:0@12.984] global ./opencv/modules/videoio/src/cap_ffmpeg_hw.hpp (566) hw_create_device FFMPEG: Created video acceleration context (av_hwdevice_ctx_create) for cuda on device 'default'
    [ INFO:0@12.984] global ./opencv/modules/core/src/ocl.cpp (1186) haveOpenCL Initialize OpenCL runtime...
    [ INFO:0@12.990] global ./opencv/modules/core/src/ocl.cpp (1192) haveOpenCL OpenCL: found 1 platforms
    [OPENCV:FFMPEG:24] This encoder is deprecated, use 'hevc_nvenc' instead
    [OPENCV:FFMPEG:40] Loaded Nvenc version 12.1
    [OPENCV:FFMPEG:40] Nvenc initialized successfully
    ```
