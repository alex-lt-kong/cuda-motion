# Camera server

A C++17 project inspired by
[Motion](https://github.com/Motion-Project/motion)
but with hardware-acceleration (i.e., CUDA) in mind.

There are two main functions of this program:

1.  Detect image change (i.e., "motion") from a video feed and record videos
    during the periods when a motion event occurs;
1.  Provide various inter-process communication methods for downstream
    programs to consume live images from different types of cameras
    so that downstream programs don't have to implement their own version of video
    feed handling repetitively. The following methods are currently supported:

    1.  File;
    1.  HTTP;
    1.  POSIX Shared Memory;
    1.  ZeroMQ Pub/Sub mode.

- In [Motion](https://github.com/Motion-Project/motion), hardware-acceleration
  can only be achieved by piping data to external libraries such as FFmpeg,
  and some computationally expensive tasks such as motion detection just
  can't be offloaded to a GPU. This greatly limits the practical use of
  Motion as we won't be able to handle even just a few (say, five) video feeds
  with an average CPU.

  - In contrast, this project is designed with CUDA support in mind, and it
    tries to offload all parallelable tasks to GPUs.

## Dependencies

- [Oat++](https://github.com/oatpp/) for HTTP service support:

  ```
  git clone https://github.com/oatpp/oatpp.git
  cd oatpp && mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release ../
  make -j4
  sudo make install
  ```

- `nlohmann-json3 (>= 3.9)`, JSON support: `apt install nlohmann-json3-dev`
- `spdlog` for logging: `apt install libspdlog-dev`
- `v4l-utils`: for manually examining and manipulating local video devices.
- `cppzmq` for message queue, `apt install libzmq3-dev`
- `FFmpeg` and `OpenCV`: image/video manipulation libraries that do all the
  heavy lifting.

  - Check build notes [here](./helper/build-notes.md) to build FFmpeg and
    OpenCV properly.

## Build and deployment

-

```bash
mkdir ./build
cmake ../
make -j2
```

- Prepare configuration file:
  - Copy `./configs/camera-server.jsonc` to the default location,
    `$HOME/.configs/ak-studio/camera-server.jsonc`; or
  - Start `./cs` with the path of configuration file manually, e.g.,
    `./cs /tmp/cs.jsonc`.

## Quality assurance

- Instead of `cmake ../`, run:

  - `cmake -DBUILD_ASAN=ON ../`
  - `cmake -DBUILD_UBSAN=ON ../`
  - `cmake -DBUILD_THSAN=ON -DCMAKE_CXX_COMPILER=clang++ ../`

  to turn on different sanitizers.

- The repo is also tested with `Valgrind` from time to time:
  `valgrind --leak-check=yes --log-file=valgrind.rpt ./build/cs`.
