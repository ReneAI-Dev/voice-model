#!/bin/bash

# Download the Mozilla TTS repository as a ZIP file
curl -LOk https://github.com/mozilla/TTS/archive/refs/heads/main.zip

# Unzip the downloaded file
unzip main.zip

# Delete the downloaded ZIP file
rm main.zip

# run your preprocessing script
python preprocess.py

# directory containing the TTS code
# TTS-main

# run the training script with your configuration file
python TTS-main/TTS/bin/train_tacotron.py --config_path config.json
