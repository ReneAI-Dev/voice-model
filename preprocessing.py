import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Directory containing .wav files
wav_dir = './'

# CSV file containing transcriptions
transcriptions_file = 'transcriptions.csv'
if not os.path.isfile(transcriptions_file):
    # Create an empty DataFrame with the appropriate columns
    default_transcriptions = pd.DataFrame(columns=['filename', 'transcription'])
    # Write the DataFrame to a CSV file
    default_transcriptions.to_csv(transcriptions_file, index=False)

# Load transcriptions
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
        print(f"File not found: {wav_file}")
        continue

    # Load the audio
    audio = AudioSegment.from_wav(wav_file)

    # Remove silence
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    # Join chunks back into a single audio segment
    audio_without_silence = AudioSegment.empty()
    for chunk in chunks:
        audio_without_silence += chunk

    # Convert back to numpy array and resample
    y = np.array(audio_without_silence.get_array_of_samples())
    y_resampled = librosa.resample(y, audio.frame_rate, 16000)

    # Compute mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y_resampled, sr=16000)

    # Encode the transcription
    encoding = [ord(c) for c in transcription]

    # Append the preprocessed audio and transcription to our lists
    file_paths.append(wav_file)
    transcript_texts.append(transcription)
    spectrograms.append(spectrogram)
    encodings.append(encoding)

# Convert the lists to a DataFrame
dataset = pd.DataFrame({
    'filename': file_paths,
    'transcription': transcript_texts,
    'spectrogram': spectrograms,
    'encoding': encodings
})

# Save the dataset to a new CSV file
dataset.to_csv('preprocessed_dataset.csv', index=False)
