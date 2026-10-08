#!/bin/sh
# encode.sh <frames_dir> <master.wav> <out.mp4>
set -e
ffmpeg -hide_banner -loglevel error -y -framerate 60 -i "$1/f_%05d.png" -i "$2" \
  -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -profile:v high -tune film \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart "$3"
