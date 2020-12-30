# Pytorch for Raspberry Pi Bash Install Script
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
sudo apt install zip unzip

echo $PATH
export PATH=/home/pi/.local/bin:$PATH

pip install gdown

sudo cp ~/.local/bin/gdown /usr/local/bin/gdown

# download the pre-compiled build
gdown https://drive.google.com/uc?id=1NAQ1nKvZmcun6isP1Bc8X434VLJJZu2J

unzip pytorch.zip

cd ./pytorch/

sudo -E python3 setup.py install

cd ..
sudo apt-get install libopenblas-dev libblas-dev m4 cmake cython \
  python3-dev python3-yaml python3-setuptools

sudo -E python3 setup.py install

cd ../