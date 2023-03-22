# Useful Commands

## 1. `v4l2-ctl`

* List supported video devices: `v4l2-ctl --list-devices`
* List supported resolutions of a video device: `v4l2-ctl --list-formats-ext --device <videoUri>`
* Get pixel format from a video device: `v4l2-ctl --get-fmt-video --device <videoUri>`
* Set pixel format to `MJPG` to a video device: `v4l2-ctl --set-fmt-video=pixelformat=MJPG --device <videoUri>`
* Get all parameters `v4l2-ctl --get-parm --all --device <videoUri>`
* Set a parameter: `v4l2-ctl --set-ctrl=<parameterName>=<parameterValue>`
* Get framerate: `v4l2-ctl --get-parm --device <videoUri>`
* Set framerate: `v4l2-ctl --set-parm=30 --device <videoUri>`

## 2. `ffmpeg` and `ffprobe`

* Open a video device: `ffmpeg -f v4l2 -framerate 25 -video_size 640x480 -i /dev/video0 output.mkv`

### Ascertain the `-pix_fmt` option is `ffmpeg` command

* Seems that there isn't a credible way lol.
* Issue `ffmpeg -pix_fmts` and try the results one by one.
* Some options are more likely to be the right value than others, such as
`yuv420p`, `yuyv422`, `bgr24`, `rgb24`.

### Ascertain the FPS of a video source
```
# ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate <videoUri>
25/1
```
* Note that this value may not be accurate for remote video sources.
* A more accurate but less formal way is to simply observe the output from
`ffmpeg` itself.

