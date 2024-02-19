#!/bin/bash

# Home dir
# cd .

# Remove the 'tegridy-tools' directory if it exists
if [ -d "tegridy-tools" ]; then
  rm -rf tegridy-tools
fi

# Clone the repository
git clone --depth 1 https://github.com/asigalov61/tegridy-tools

# Check for CUDA and set the requirements and setup file paths
REQUIREMENTS_PATH="./tegridy-tools/install/requirements.txt"
SETUP_PATH="tegridy-tools/install/setup.py"
if command -v nvcc &> /dev/null
then
    echo "CUDA is installed"
    REQUIREMENTS_PATH="./tegridy-tools/install/requirements_cuda.txt"
    SETUP_PATH="tegridy-tools/install/setup_cuda.py"
else
    echo "CUDA is not installed"
fi

# Install Python packages
pip install -r $REQUIREMENTS_PATH

# Install fluidsynth
apt install fluidsynth

# Run the Python script
python $SETUP_PATH

# Return to the home directory
# cd .