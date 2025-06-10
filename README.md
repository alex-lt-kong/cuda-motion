# Cuda motion

A C++20 project inspired by
[Motion](https://github.com/Motion-Project/motion)
but is GPU-accelerated.

There are two main functions of this program:

1.  Detect image change (i.e., "motion") from a video feed and record videos
    during the periods when a motion event occurs;
1.  Provide various inter-process communication methods for downstream
    programs to consume live images from different types of cameras
    so that downstream programs don't have to implement their own version of
    video feed handling repetitively. The following methods are currently
    supported:

    1.  File;
    1.  HTTP;
    1.  POSIX Shared Memory;
    1.  ZeroMQ Pub/Sub mode (with ProtoBuf encoding).

- In [Motion](https://github.com/Motion-Project/motion), hardware acceleration
  can only be achieved by piping data to external libraries such as FFmpeg,
  and some computationally expensive tasks such as motion detection just
  can't be offloaded to a GPU (as it is beyond the capability of FFmpeg).
  This greatly limits the practical use of Motion as we won't be able to
  handle even just a few (say, five) video feeds with an average CPU.

  - In contrast, this project is designed to work only if a CUDA-compatible
    device (e.g., an Nvidia GPU) is in place, and it offloads all
    parallelable tasks to GPUs, achieving much better performace.

  - On the con side, [Motion](https://github.com/Motion-Project/motion) works
    as long as you have a CPU😉, but
    [Cuda motion](https://github.com/alex-lt-kong/cuda-motion) can only run if
    you have an Nvidia GPU and its closed source CUDA framework installed.

## Dependencies

- [vcpkg.json](./vcpkg.json)
- `v4l-utils`: for manually examining and manipulating local video devices.
- `OpenCV`: image/video manipulation libraries that do all the
  heavy lifting.
  - Check build notes [here](./helper/build-notes.md) and
    [here](https://github.com/alex-lt-kong/the-nitty-gritty/tree/main/c-cpp/cpp-only/06_poc/05_cuda-vs-ffmpeg).

## Build and deployment

```bash
mkdir ./build
cmake ../
make -j2
```

## Quality assurance

- Instead of `cmake ../`, run:

  - `cmake -DBUILD_ASAN=ON ../`
  - `cmake -DBUILD_UBSAN=ON ../`

- The repo is also tested with `Valgrind` from time to time:
  `valgrind --leak-check=yes --log-file=valgrind.rpt ./build/cs`.

## Profiling

- gprof:

```bash
cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg  ../
make -j4
./build/cm
gprof ./build/cm gmon.out
```

- callgrind

```
valgrind --tool=callgrind ./cm
kcachegrind `ls -tr callgrind.out.* | tail -1`
```
