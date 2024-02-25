# Profiling

## GNU gprof

- Enable profiling: `cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg ..`

  - Then build as usual: `make -j2`

- Run `./cs` and exit it normally. A `gmon.out` file will be generated.

- Run `gprof` to analyze `gmon.out` to get human-readable result: `gprof ./cs > ./gmon.rpt`

- Unfortunately, gprof is only able to catch around 20% of CPU time with the
  remaining 80% missing.

## Valgrind

- Just run `valgrind --tool=callgrind ./cs`

- Unfortunately, `valgrind` greatly slows down the program to the extent that
  the program does not appear to be in the normal state, rendering the data
  collected most probably not realistic.

## Perf

- `apt install linux-tools-common linux-tools-generic`

- Collect data: `perf record ./cs`

  - Exit `./cs` with `ctrl-C` to finish data collection step.

- Report data being collected: `perf report`

- The report does not seem to be complete as it only captures around 10%
  of CPU time. It is likely that `perf` can collect data from the main thread
  only.

## [Poor man's profiler](http://poormansprofiler.org/)

- While not strictly a profiler, this trick provides some very useful insight.

- Build Cuda Motion will debug symbols: `make -DCMAKE_BUILD_TYPE=RelWithDebInfo ..`

  - And just run it: `./cs`

- Get call stacks of all threads:
  `gdb -ex "set pagination 0" -ex "thread apply all bt" --batch -p <pid>`

- Samples of a heavily loaded thread, which reveals quite a bit of useful
  information:

      *
      ```
      Thread 10 (Thread 0x7f7ee67fc640 (LWP 2713) "cs"):
      #0  0x000055c338579a7d in cv::flipHoriz(unsigned char const*, unsigned long, unsigned char*, unsigned long, cv::Size_<int>, unsigned long) ()
      #1  0x000055c33857bcdf in cv::flip(cv::_InputArray const&, cv::_OutputArray const&, int) ()
      #2  0x000055c3384d6d45 in deviceManager::InternalThreadEntry (this=0x55c339b36850) at ./camera-server/src/camera-server/device_manager.cpp:853
      #3  0x000055c33844aefe in MyEventLoopThread::InternalThreadEntryFunc (This=<optimized out>) at ./camera-server/src/camera-server/event_loop.h:60
      #4  0x00007f7eef1f4b43 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
      #5  0x00007f7eef286a00 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
      ```

      *
      ```
      Thread 10 (Thread 0x7f5cfcffe640 (LWP 3236) "cs"):
      #0  0x00007f5d0160a603 in ?? () from /usr/local/lib/libswscale.so.5
      #1  0x00007f5d015f76c7 in ?? () from /usr/local/lib/libswscale.so.5
      #2  0x00007f5d015e0049 in ?? () from /usr/local/lib/libswscale.so.5
      #3  0x00007f5d015e1147 in sws_scale () from /usr/local/lib/libswscale.so.5
      #4  0x000055cfaf9525b6 in CvCapture_FFMPEG::retrieveFrame(int, unsigned char**, int*, int*, int*, int*) ()
      #5  0x000055cfaf95297b in cv::(anonymous namespace)::CvCapture_FFMPEG_proxy::retrieveFrame(int, cv::_OutputArray const&) ()
      #6  0x000055cfaf9011f8 in cv::VideoCapture::retrieve(cv::_OutputArray const&, int) ()
      #7  0x000055cfaf900b44 in cv::VideoCapture::read(cv::_OutputArray const&) ()
      #8  0x000055cfaf44fa04 in deviceManager::InternalThreadEntry (this=0x55cfb175a850) at ./camera-server/src/camera-server/device_manager.cpp:823
      #9  0x000055cfaf3c3efe in MyEventLoopThread::InternalThreadEntryFunc (This=<optimized out>) at ./camera-server/src/camera-server/event_loop.h:60
      #10 0x00007f5d010bdb43 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
      #11 0x00007f5d0114fa00 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
      ```

      *
      ```
      Thread 10 (Thread 0x7fd589a3d640 (LWP 3644) "cs"):
      #0  0x00007fd571c41235 in ?? () from /lib/x86_64-linux-gnu/libnvcuvid.so.1
      #1  0x00007fd58ef8ce79 in ?? () from /usr/local/lib/libavcodec.so.58
      #2  0x00007fd571c16d58 in ?? () from /lib/x86_64-linux-gnu/libnvcuvid.so.1
      #3  0x00007fd571c70571 in ?? () from /lib/x86_64-linux-gnu/libnvcuvid.so.1
      #4  0x00007fd571c6e39e in ?? () from /lib/x86_64-linux-gnu/libnvcuvid.so.1
      #5  0x00007fd571c1572b in ?? () from /lib/x86_64-linux-gnu/libnvcuvid.so.1
      #6  0x00007fd58ef8d0e1 in ?? () from /usr/local/lib/libavcodec.so.58
      #7  0x00007fd58ef8dc02 in ?? () from /usr/local/lib/libavcodec.so.58
      #8  0x00007fd58efbd14f in ?? () from /usr/local/lib/libavcodec.so.58
      #9  0x00007fd58efbdfc8 in avcodec_send_packet () from /usr/local/lib/libavcodec.so.58
      #10 0x000055f74572f20b in CvCapture_FFMPEG::grabFrame() [clone .part.0] ()
      #11 0x000055f7456e1f52 in cv::VideoCapture::grab() ()
      #12 0x000055f7456e1b31 in cv::VideoCapture::read(cv::_OutputArray const&) ()
      #13 0x000055f745230a04 in deviceManager::InternalThreadEntry (this=0x55f747524850) at ./camera-server/src/camera-server/device_manager.cpp:823
      #14 0x000055f7451a4efe in MyEventLoopThread::InternalThreadEntryFunc (This=<optimized out>) at ./camera-server/src/camera-server/event_loop.h:60
      #15 0x00007fd58e2fdb43 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
      #16 0x00007fd58e38fa00 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
      ```
      *
      ```
      Thread 10 (Thread 0x7fd589a3d640 (LWP 3644) "cs"):
      #0  0x000055f7454a3f70 in cv::FillConvexPoly(cv::Mat&, cv::Point_<long> const*, int, void const*, int, int) ()
      #1  0x000055f7454aa1b2 in cv::PolyLine(cv::Mat&, cv::Point_<long> const*, int, bool, void const*, int, int, int) ()
      #2  0x000055f7454aafee in cv::putText(cv::_InputOutputArray const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, cv::Point_<int>, int, double, cv::Scalar_<double>, int, int, bool) ()
      #3  0x000055f745223c88 in deviceManager::overlayStats (this=0x55f747524850, frame=..., changeRate=<optimized out>, cooldown=<optimized out>, videoFrameCount=<optimized out>) at ./camera-server/src/camera-server/device_manager.cpp:453
      #4  0x000055f7452308d2 in deviceManager::InternalThreadEntry (this=0x55f747524850) at ./camera-server/src/camera-server/device_manager.cpp:877
      #5  0x000055f7451a4efe in MyEventLoopThread::InternalThreadEntryFunc (This=<optimized out>) at ./camera-server/src/camera-server/event_loop.h:60
      #6  0x00007fd58e2fdb43 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
      #7  0x00007fd58e38fa00 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
      ```

## References

1. [gprof Quick-Start Guide"][1]

[1]: https://web.eecs.umich.edu/~sugih/pointers/gprof_quick.html "gprof Quick-Start Guide"
