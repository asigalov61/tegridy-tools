# Tensorflow 2.0 for Raspberry Pi Bash Install Script
# Version 1.0
#
# Put this script to the /home/pi/dir
#
# Project Los Angeles
# Tegridy Code 2020
##############################################
cd /home/pi/

sudo apt-get update
sudo apt-get upgrade

sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow

sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev cython
sudo -H pip3 install pybind11
sudo -H pip3 install h5py==2.10.0

sudo -H pip3 install --upgrade setuptools

echo $PATH
export PATH=/home/pi/.local/bin:$PATH

pip install gdown
sudo cp ~/.local/bin/gdown /usr/local/bin/gdown

gdown https://drive.google.com/uc?id=1rYIbenjV1wV7D0AbjlZ9z-NKPq1yi3BG

sudo -H pip3 install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl