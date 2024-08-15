#!/bin/bash

mkdir ./tmp
wget -O tmp/metric_pretrained.zip "https://www.dropbox.com/scl/fi/b8gl1w5eotn498yn3tdjl/metric_pretrained.zip?rlkey=gjrj0izhycmj1imloeh3nsv8r&st=ilmbf8he&dl=1"
unzip tmp/metric_pretrained.zip -d tmp/

mv tmp/metric_pretrained/jx_vit_base_p16_224-80ecf9dd.pth metrics/egovlp/pretrained/
mv tmp/metric_pretrained/distilbert-base-uncased metrics/egovlp/pretrained/
mv tmp/metric_pretrained/egovlp.pth metrics/egovlp/pretrained/
mv tmp/metric_pretrained/epic_mir_plus.pth metrics/egovlp/pretrained/
mv tmp/metric_pretrained/model_base_caption_capfilt_large.pth metrics/blip/pretrained/
mv tmp/metric_pretrained/model_large_caption.pth metrics/blip/pretrained/

rm -rf ./tmp
