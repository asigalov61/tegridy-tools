#!/bin/bash

# Clone the repository
git clone --depth 1 https://github.com/asigalov61/tegridy-tools

# Change directory to the cloned repository
cd tegridy-tools/install

# Install Python packages
pip install -r requirements.txt

# Install fluidsynth
apt install fluidsynth

# Run the Python script
python setup.py

# Return to the home directory
cd