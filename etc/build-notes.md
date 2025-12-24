# Build FFmpeg and OpenCV for Cuda Motion

- This document is a revised version of the README of
  [this PoC](https://github.com/alex-lt-kong/proofs-of-concept/tree/main/cpp/05_cuda-vs-ffmpeg)

- To leverage GPU with OpenCV, there are two options, either we build OpenCV
  with `cv::cudacodec` or we build OpenCV
  with FFmpeg (`avcodec`/`avformat`/etc) with GPU support. Or we can do both in
  one go, i.e., we firstly build ffmpeg
  with GPU support, and then we build OpenCV with both GPU-enabled FFmpeg and
  `cv::cudacodec`

## Prepare Nvidia GPU infrastructure

1. Make sure you have a compatible GPU of course.

    - Nvidia provides a support matrix
      [here](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)

2. Install Nvidia's GPU driver
   [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#package-manager)

3. Package `nvidia-utils-<version>` should be automatically installed by the
   above steps. If so, try `nvidia-smi` to see the status of your GPU. `nvtop`
   is a usefull third-party tool that demonstrate more GPU details
   in a manner similar to `htop`: `apt install nvtop`.

## OpenCV with FFmpeg (`avcodec`/`avformat`/etc) that enables CUDA

### I. Build FFmpeg

1. While OpenCV supports multiple backends, it seems that FFmpeg enjoys
   better support from Nvidia, making it a good option to work with Nvidia's
   GPU.

1. If there is an `FFmpeg` installed by `apt`, remove it first.

    - Ubuntu does allow multiple versions to co-exist, the issue is that
      in compilation/linking, it make not be easy to configure different build
      systems (CMake/Ninja/handwritten ones) to use exactly the version of
      FFmpeg we want. Therefore, it is simpler if we can remove other
      unused FFmpeg.
    - Apart from `FFmpeg` itself, it is possible that some of its libraries,
      such as `libswscale` and `libavutil` can exist independently. If so, try
      commands such as `find / -name libswscale.so* 2> /dev/null` to find them
      and then issue commands such as `apt remove libswscale5` to remove them.

1. Build `FFmpeg` with Nvidia GPU support **based on** this
   [Nvidia document](https://docs.nvidia.com/video-technologies/video-codec-sdk/pdf/Using_FFmpeg_with_NVIDIA_GPU_Hardware_Acceleration.pdf)
   and
   this [FFmpeg document on HWAccel](https://trac.ffmpeg.org/wiki/HWAccelIntro).

- We may need to combine options gleaned from different sources to
  construct the final `./configure` command.

1. One working version of `./configure` is:
   `./configure --enable-pic --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec --extra-cflags="-I/usr/local/cuda/include" --extra-cflags="-fPIC" --extra-ldflags=-L/usr/local/cuda/lib64  --nvccflags="-gencode arch=compute_75,code=sm_75 -O2"`

    -
   `--enable-cuda-llvm --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec`:
   enable Nvidia Video Codec
   SDK
   ([source](https://trac.ffmpeg.org/wiki/HWAccelIntro))
    - `--enable-pic --enable-shared --extra-cflags="-fPIC"`, used to solve
      the issue during OpenCV build in a later stage: "/usr/bin/ld:
      /usr/local/lib/libavcodec.a(vc1dsp_mmx.o):
      relocation R_X86_64_PC32 against symbol `ff_pw_9' can not be used when
      making a shared object; recompile with -fPIC"
      ([source1](https://www.twblogs.net/a/5ef71a3c209c567d16133dae),
      [source2](https://askubuntu.com/questions/1292968/error-when-installing-opencv-any-version-on-ubuntu-18-04))

1. It is likely that we need to compile FFmpeg multiple times to have the
   desired functionalities, to not mess the source directory, it is recommended
   that we build the project "out of tree":

```bash
mkdir /tmp/FFmpeg
cd /tmp/FFmpeg
~/repos/FFmpeg/configure <whatever arguments>
```

1. If `./configure`'s report contains strings `h264_nvenc`/`hevc_nvenc`/etc
   like below, the proper hardware encoders/decoders are correctly configured:

```
Enabled encoders:
...
adpcm_argo              aptx                    dvbsub                  h264_nvenc              msmpeg4v2               pcm_s16le_planar        pcm_u8                  roq                     targa                   wmav2
adpcm_g722              aptx_hd                 dvdsub                  h264_v4l2m2m            msmpeg4v3               pcm_s24be               pcm_vidc                roq_dpcm                text                    wmv1
adpcm_g726              ass                     dvvideo                 hevc_nvenc              msvideo1                pcm_s24daud             pcx                     rpza                    tiff                    wmv2
...
Enabled hwaccels:
av1_nvdec               hevc_nvdec              mpeg1_nvdec             mpeg4_nvdec             vp8_nvdec               wmv3_nvdec
h264_nvdec              mjpeg_nvdec             mpeg2_nvdec             vc1_nvdec               vp9_nvdec

```

1. If build is successful, execute `FFmpeg` and check if it works.

    - It should show something like:

   ```
   ffmpeg version n4.4.3-48-gc3ad886251 Copyright (c) 2000-2022 the FFmpeg developers
   built with gcc 11 (Ubuntu 11.3.0-1ubuntu1~22.04)
   configuration: --enable-pic --enable-shared --enable-nonfree --enable-cuda-sdk --enable-cuda-llvm --enable-ffnvcodec --enable-cuvid --enable-nvenc --enable-nvdec --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-cflags=-fPIC --extra-ldflags=-L/usr/local/cuda/lib64
   libavutil      56. 70.100 / 56. 70.100
   libavcodec     58.134.100 / 58.134.100
   ...
   ```

    - If Nvidia's GPU acceleration is compiled in, issuing
      `ffmpeg -codecs | grep h264` should show something like:

   ```
   ...
   DEV.LS h264                 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (decoders: h264 h264_v4l2m2m h264_cuvid ) (encoders: h264_nvenc h264_v4l2m2m nvenc nvenc_h264 )
   ```

### II. Build OpenCV

1. For OpenCV build steps, refer to the next section, here are just some
   comments on the FFmpeg part of the OpenCV build

1. The first perennial problem when building OpenCV on top of a custom FFmpeg
   build is compatibility--each OpenCV version depends on one or few snapshots
   of
   FFmpeg versions, if the versions of OpenCV and FFmpeg mismatch, various
   compilation errors would occur. \* After multiple attempts, this note is
   prepared with FFmpeg `n4.4.3`
   and OpenCV `4.6.0`.

1. Issue
   `cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX="$HOME/.local"  ..`
   to install the built OpenCV to the user's home directory. \* Verify the
   result of `cmake` carefully before proceeding
   to `make`,
   make sure that `cmake` shows the expected version of `FFmpeg`.

1. If OpenCV fails to compile due to a compatibility issue with FFmpeg, we need
   to go back to `I` and build FFmpeg again. \* Before issuing
   `git checkout <branch/tag>`, one needs to issue
   `sudo make uninstall` and `sudo make distclean` to make sure the previous
   FFmpeg build is cleared; otherwise multiple FFmpeg versions could co-exist,
   causing confusion and strange errors.

## III. Set environment variables

- TL;DR: add the below to `~/.profile`/`~/.bashrc`

```
export OPENCV_VIDEOIO_DEBUG=0
export OPENCV_FFMPEG_DEBUG=0
export OPENCV_LOG_LEVEL=DEBUG
export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;cuvid|video_codec;h264_cuvid"
#export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;cuvid|video_codec;mjpeg_cuvid"
export OPENCV_FFMPEG_WRITER_OPTIONS="hw_encoders_any;cuda"
```

1. Enable OpenCV's debug logging, which could reveal useful information:

   ```bash
   export OPENCV_VIDEOIO_DEBUG=1
   export OPENCV_FFMPEG_DEBUG=1
   export OPENCV_LOG_LEVEL=DEBUG
   ```

1. Enable hardware decoding for `cv::VideoCapture` with environment variable
   `OPENCV_FFMPEG_CAPTURE_OPTIONS`. This envvar uses a special key/value pair
   format `key1;val1|key2;val2`. To use hardware decoder, we want to set it
   to something like:

- `"hwaccel;cuvid|video_codec;mjpeg_cuvid"`--if we know the video feed is in
  MJPG format.
- `"hwaccel;cuvid|video_codec;h264_cuvid"`--if we know the video feed is in
  h264 format.
- List the supported codec by `ffmpeg -codecs | grep nv`.

1. Enable hardware decoding for `cv::VideoWriter` with environment variable
   `OPENCV_FFMPEG_WRITER_OPTIONS`. This envvar uses a special key/value pair
   format `key1;val1|key2;val2`. To use hardware encoder, we want to set it
   to `"hw_encoders_any;cuda"` \* It appears that OpenCV does not allow us to
   pick a specific encoder.

## OpenCV with `cv::cudacodec`

- Have Nvidia GPU, driver and CUDA library properly installed (verify this
  by issuing `nvidia-smi`)

- Download Nvidia's Video Codec SDK. This is a very confusing step as seems
  it is documented nowhere except in a stack exchange post
  [here](https://stackoverflow.com/questions/65740367/reading-a-video-on-gpu-using-c-and-cuda)

    - Long story short, we need to download Nvidia's Video Codec SDK
      [here](https://developer.nvidia.com/video-codec-sdk) and copy all header
      files
      in `./Interface/` directory to corresponding CUDA's include directory
      (e.g., `/usr/local/cuda/targets/x86_64-linux/include/` or any directory
      that
      follows the `-- NVCUVID: Header not found` complaint)

    - Without this step, `OpenCV`'s `cmake`/`make` could still work, but the
      compiled code will complain:

      ```
      terminate called after throwing an instance of 'cv::Exception'
      what():  OpenCV(4.7.0) ./repos/opencv/modules/core/include/opencv2/core/private.cuda.hpp:112: error: (-213:The function/feature is not implemented) The called functionality is disabled for current build or platform in function 'throw_no_cuda'
      ```

    - It is also noteworthy that the so called "Nvidia's Video Codec SDK" is the
      same as the "nv-codec-headers.git" mentioned in the
      [HWAccelIntro](https://trac.ffmpeg.org/wiki/HWAccelIntro)
      of the OpenCV with FFmpeg route documented below. However, FFmpeg states
      that it uses a slightly modified version of Nvidia Video Codec SDK and
      experiment also shows that the headers installed by `nv-codec-headers`
      won't
      be recognized by OpenCV's `cmake`. So we need two copies of the same set
      of
      headers files for two routes to work concurrently.

- Prepare [opencv_contrib](https://github.com/opencv/opencv_contrib) repository.
  OpenCV needs it to build `cuda` support.

- The final `cmake` command should look like below:

```bash    
  # export INSTALL_PATH="$HOME/opt/opencv-$OPENCV_VERSION"  # export INSTALL_PATH="$HOME/opt/opencv-$OPENCV_VERSION"
  export INSTALL_PATH=/usr/local 
  mkdir -p "$INSTALL_PATH"
  # reveal the CUDA architecture and we build for it only
  export CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=noheader,csv | tail -n1)
  # Find my own FFmpeg
  export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
  # ts: test support
  # cudafilters: a test dependency for cudaimgproc
  # dnn: deep neural network support
  # objdetect: for facial detection and recognition
  # cudaarithm: for cv::cuda::addWeighted call
  export BUILD_LIST="core,imgproc,videoio,imgcodecs,cudev,cudacodec,ts,cudaarithm,cudaimgproc,cudawarping,cudafilters,dnn,objdetect,cudaarithm"

  cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_LIST="${BUILD_LIST}" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
    -D WITH_CUDA=ON \
    -D WITH_NVCUVID=ON \
    -D WITH_NVCUVENC=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/repos/opencv_contrib/modules/ \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME="opencv.pc" \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_GSTREAMER=ON \
    -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
    -D OPENCV_TEST_DATA_PATH=~/repos/opencv_extra/testdata \
    ..
```

and the `cmake` output should show lines that indicate the inclusion of
NVCUVID / NVCUVENC:

```
-- Found NVCUVID: /usr/lib/x86_64-linux-gnu/libnvcuvid.so
-- Found NVCUVENC:  /usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-encode.so
...
-- NVIDIA CUDA:                   YES (ver 11.8, CUFFT CUBLAS NVCUVID NVCUVENC)
-- NVIDIA GPU arch:             86
-- NVIDIA PTX archs:
```

as well as FFmpeg:

```
-- Video I/O:
-- FFMPEG:                      YES
-- avcodec:                   YES (61.19.101)
-- avformat:                  YES (61.7.100)
-- avutil:                    YES (59.39.100)
-- swscale:                   YES (8.3.100)
-- avresample:                NO
```

The same information should be printed when `cv::getBuildInformation()` is called.

- Build step: `cmake --build . --parallel $(nproc) --config Release`
    - `--config Release` should only be needed on Windows

- `ctest`:

```bash
# many tests expect testdata to work
export OPENCV_TEST_DATA_PATH=$HOME/repos/opencv_extra/testdata
# Disable multi-GPU setup, seems OpenCV's test suites do not support it
export CUDA_VISIBLE_DEVICES=0
ctest --exclude-regex "opencv_test_highgui|hal_intrin128" -j$(nproc) --rerun-failed --output-on-failure
```

- `sudo cmake --install .`