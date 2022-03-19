# Useful Commands

## How to determine the `-pix_fmt` option is `ffmpeg` command

* Seems that there isn't a credible way lol.
* Issue `ffmpeg -pix_fmts` and try the results one by one.
* Some options are more likely to be the right value than others, such as `yuv420p`, `yuyv422`, `rgb24`.