#!/usr/bin/env bash

# Script to download the wildfire photo dataset provided by Pyronear.
#
# It was produced with the following repo:
# https://github.com/pyronear/pyro-dataset

set -x

cd data/01_raw || exit

gdown --fuzzy https://drive.google.com/file/d/17syKwltw8Jv-nYlLjzxbXQDd4p9hmkKC/view?usp=sharing

rm -rf DS-71c1fd51-v2
unzip DS-71c1fd51-v2.zip
rm DS-71c1fd51-v2.zip
