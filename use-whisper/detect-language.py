import whisper
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from operator import itemgetter

import tkinter as tk
from tkinter import filedialog
from pathlib import Path 


#standard OpenAI models 
model = whisper.load_model("tiny") #turbo #tiny #"large-v3"

root = tk.Tk()
root.withdraw()

# Open the file dialog and get the path
file_path = filedialog.askopenfilename(
    title="Select a File",
    initialdir="d:/code",
    filetypes=(("Audio files", "*.mp3 *.wav"), ("All files", "*.*"))
)
print(f"Selected file: {file_path}")

# load audio and pad/trim it to fit 30 seconds
# Initialize and hide the main window



audio = whisper.load_audio(file_path) #sanskrit audio
# audio = whisper.load_audio("Atharvaveda_Kanda_10_0001.wav") #sanskrit audio
# audio = whisper.load_audio("bg-chap1.mp3") #background audio
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
tenv, probs = model.detect_language(mel)
# _, probs = model.detect_language(mel)
probs_sorted = sorted(probs.items(),  key=itemgetter(1), reverse=True) # type: ignore
# probs_sorted = sorted(probs,key=itemgetter(1), reverse=True) # type: ignore
# probs_sorted = sorted(probs,key=probs.get, reverse=True) # type: ignore
print(f"Detected language: {max(probs, key=probs.get)}") # type: ignore

# output the language probabilities :ss-audio-may-23.mp3 
    # ('si', 0.5062861442565918) : Sinhala
    # ('hi', 0.10614561289548874) : Hindi
    # ('pa', 0.042809389531612396) : Punjabi
    # ('en', 0.04274876415729523) : English
    # ('kn', 0.04115227237343788) : Kannada
    # ('sa', 0.03515150025486946) : Sanskrit


# output the language probabilities :bg-chap1.mp3 
    # ('sa', 0.22502821683883667) : Sanskrit
    # ('en', 0.16907507181167603) : English
    # ('lv', 0.10124222934246063) : Latvian

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)