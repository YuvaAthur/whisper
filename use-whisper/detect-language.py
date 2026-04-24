import whisper
from operator import itemgetter


model = whisper.load_model("tiny") #turbo 

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("ss-audio-may-23.mp3") #sanskrit audio
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

# output the language probabilities
    # ('si', 0.5062861442565918) : Sinhala
    # ('hi', 0.10614561289548874) : Hindi
    # ('pa', 0.042809389531612396) : Punjabi
    # ('en', 0.04274876415729523) : English
    # ('kn', 0.04115227237343788) : Kannada
    # ('sa', 0.03515150025486946) : Sanskrit


# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)