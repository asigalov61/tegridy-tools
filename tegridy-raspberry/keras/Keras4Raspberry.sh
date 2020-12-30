# Keras 2.3.0 for Raspberry Pi Bash Install Script
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

sudo apt-get install python3-h5py
pip3 install keras==2.3.0
