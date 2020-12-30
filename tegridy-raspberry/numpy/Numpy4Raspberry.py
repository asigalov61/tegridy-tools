# Numpy 1.19.4 for Raspberry Pi Bash Install Script
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

sudo apt-get install libatlas-base-dev
sudo pip3 install numpy==1.19.4