#!/bin/bash

cd `pwd`/data/custom_data/raw_files
mkdir -p audio_files

for name in *.wav;
do
    ffmpeg -i "$name" -c:a copy "../audio_files/$name" -y
done

