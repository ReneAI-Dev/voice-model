#!/bin/bash

# navigate to the directory containing your preprocessing script
cd /path/to/preprocessing

# run your preprocessing script
python preprocess.py

# navigate to the directory containing the TTS code
cd /path/to/TTS

# run the training script with your configuration file
python TTS/bin/train.py --config_path /path/to/config.json
