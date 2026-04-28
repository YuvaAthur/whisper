
# Experiment: Fine Tuned Models
Ref: detect-language-custom.py
Model: GitaWhisper from Huggingface
Code: Claude generated

Setup:
Install dependencies first:

pip install -U openai-whisper
pip install transformers torch librosa accelerate
pip install huggingface_hub

How it works
StepWhat happensLoadlibrosa decodes the MP3 and resamples to 16 kHz mono (Whisper's required format)Feature extractionThe processor converts raw audio into a log-mel spectrogramLanguage detectionThe encoder + decoder run one step; the first predicted token is always a language token (e.g. <|en|>, <|hi|>, <|sa|>)ScoringSoftmax over all language tokens gives a probability for each languageTranscriptionOptional full transcription using the detected language

Setting up HuggingFace (HF)
`pip install huggingface_hub
`hf scan-cache

## Outputs: Model GitaWhisper-Tiny
`---- DIABOLIC6045/GITAWHISPER-TINY ----
load_audio("ss-audio-may-23.mp3") #sanskrit audio 
    # Top 5 Languages:
    #    cs             0.01%
    #    tl             0.01%
    #    th             0.01%
    #    ta             0.00%
    #    hi             0.00%
load_audio("Atharvaveda_Kanda_10_0001.wav") #sanskrit audio
    # Top 5 Languages:
    #    kn             0.01%
    #    ta             0.01%
    #    en             0.01%
    #    te             0.00%
    #    sa             0.00%
load_audio("bg-chap1.mp3") #background audio
    # Top 5 Languages:
    #    ta             0.05%
    #    en             0.02%
    #    hi             0.01%
    #    sa             0.01%
    #    te             0.01%