import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import requests

def log(text):
    print(text)
    discord(text)

def discord(content):
    webhook_url = 'url_here'
    requests.post(webhook_url, data={'content': content})

# Directory containing .wav files
wav_dir = './'

# CSV file containing transcriptions
transcriptions_file = 'transcriptions.csv'
if not os.path.isfile(transcriptions_file):
    # Create an empty DataFrame with the appropriate columns
    default_transcriptions = pd.DataFrame(columns=['filename', 'transcription'])
    # Write the DataFrame to a CSV file
    default_transcriptions.to_csv(transcriptions_file, index=False)
    log("Created a new transcriptions file as it did not exist.")

# Load transcriptions
log("Loading transcriptions...")
transcriptions = pd.read_csv(transcriptions_file)

# Initialize lists to hold file paths, transcriptions, spectrograms and encodings
file_paths = []
transcript_texts = []
spectrograms = []
encodings = []

for index, row in transcriptions.iterrows():
    filename = row['filename']
    transcription = row['transcription']

    # Ensure the corresponding .wav file exists
    wav_file = os.path.join(wav_dir, filename + '.wav')
    if not os.path.isfile(wav_file):
        log(f"File not found: {wav_file}")
        continue

    log(f"Processing file: {wav_file}")
    
    # Load the audio
    audio = AudioSegment.from_wav(wav_file)

    # Remove silence
    log("Removing silence...")
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    # Join chunks back into a single audio segment
    audio_without_silence = AudioSegment.empty()
    for chunk in chunks:
        audio_without_silence += chunk

    # Convert back to numpy array and resample
    log("Converting to numpy array and resampling...")
    y = np.array(audio_without_silence.get_array_of_samples())
    y_resampled = librosa.resample(y, audio.frame_rate, 16000)

    # Compute mel-spectrogram
    log("Computing mel-spectrogram...")
    spectrogram = librosa.feature.melspectrogram(y_resampled, sr=16000)

    # Encode the transcription
    log("Encoding transcription...")
    encoding = [ord(c) for c in transcription]

    # Append the preprocessed audio and transcription to our lists
    log("Appending to the dataset...")
    file_paths.append(wav_file)
    transcript_texts.append(transcription)
    spectrograms.append(spectrogram)
    encodings.append(encoding)

# Convert the lists to a DataFrame
log("Converting to DataFrame...")
dataset = pd.DataFrame({
    'filename': file_paths,
    'transcription': transcript_texts,
    'spectrogram': spectrograms,
    'encoding': encodings
})

# Save the dataset to a new CSV file
log("Saving the dataset to preprocessed_dataset.csv...")
dataset.to_csv('preprocessed_dataset.csv', index=False)

log("Finished preprocessing the audio data.")
