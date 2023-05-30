#!/bin/bash

# navigate to the directory containing your preprocessing script
cd ./

# Download the Mozilla TTS repository as a ZIP file
curl -LOk https://github.com/mozilla/TTS/archive/refs/heads/main.zip

# Unzip the downloaded file
unzip main.zip

# Delete the downloaded ZIP file
rm main.zip

# run your preprocessing script
python preprocess.py

# navigate to the directory containing the TTS code
cd TTS-main

# run the training script with your configuration file
python TTS/bin/train.py --config_path /path/to/config.json
