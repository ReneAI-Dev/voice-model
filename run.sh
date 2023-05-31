#!/bin/bash

# Download the Mozilla TTS repository as a ZIP file
curl -LOk https://github.com/mozilla/TTS/archive/refs/heads/master.zip

# Unzip the downloaded file
unzip *.zip

# Delete the downloaded ZIP file
rm *.zip

# run your preprocessing script
python3 preprocessing.py

# directory containing the TTS code
# cd TTS-master
echo Working Directory:
pwd
echo Python Path:
echo $PYTHONPATH
# run the training script with your configuration file
python3 TTS-master/TTS/bin/train_tacotron.py --config_path config.json
