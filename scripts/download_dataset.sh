#!/bin/bash

echo Downloading dataset to $1 ...
wget -O $1/LEGO_Data.zip "https://www.dropbox.com/scl/fo/4m0v9oy753aimas8rz6v1/ANoJhZQz2BdcGIVLzUsHdP0?rlkey=o8saklcszfc098mjnpid767ic&e=1&dl=1"

echo Unzipping dataset...
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip $1/LEGO_Data.zip -d $1
rm $1/LEGO_Data.zip
unzip $1/EgoGen.zip -d $1
rm $1/EgoGen.zip

echo Done!
echo Video frames are saved in $1/EgoGen.
echo Instructions are saved in
echo $1/ego4d_train.json
echo $1/ego4d_val.json
echo $1/epickitchen_train.json
echo $1/epickitchen_val.json