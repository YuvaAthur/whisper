#Claude Code 
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── 0. TODO: Fix vulnerability before sharing ───────────────────────────────────
# ReadHFModel : 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 
import os
from huggingface_hub import login
login(token="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")
# login(token=os.getenv("HF_TOKEN")) # type: ignore

# ── 0. Simple trial - did not work ───────────────────────────────────────────────
#custom models 
# model_id = "diabolic6045/GitaWhisper-tiny"
# processor = WhisperProcessor.from_pretrained(model_id)
# model = WhisperForConditionalGeneration.from_pretrained(model_id)
# model.to("cuda" if torch.cuda.is_available() else "cpu") # type: ignore


# ── 2. Load & preprocess the MP3 ──────────────────────────────────────────────
def load_audio(mp3_path: str, target_sr: int = 16000):
    """Load an MP3 and resample to 16 kHz mono (required by Whisper)."""
    audio, sr = librosa.load(mp3_path, sr=target_sr, mono=True)
    return audio  # numpy float32 array

# ── 3. Detect language ────────────────────────────────────────────────────────
def detect_language(mp3_path: str) -> dict:
    audio = load_audio(mp3_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        # ✅ Use generate() to get the first token (language token)
        # max_new_tokens=1 forces only the language token to be predicted
        predicted_ids = model.generate(
            inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True
        )

    # The first generated token is always the language token in Whisper
    lang_token_scores = predicted_ids.scores[0]  # shape: (1, vocab_size)
    lang_probs = torch.softmax(lang_token_scores, dim=-1)

    tokenizer = processor.tokenizer

    # Whisper language tokens follow the pattern <|xx|> where xx is a 2-char lang code
    lang_tokens = {
        token: token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token.startswith("<|") and token.endswith("|>") and len(token) == 6
    }

    scored_languages = {
        token.strip("<|>"): lang_probs[0, token_id].item()
        for token, token_id in lang_tokens.items()
    }

    ranked = sorted(scored_languages.items(), key=lambda x: x[1], reverse=True)
    top_lang, top_score = ranked[0]

    return {
        "detected_language": top_lang,
        "confidence": round(top_score * 100, 2),
        "top_5": [
            {"language": lang, "probability": round(score * 100, 2)}
            for lang, score in ranked[:5]
        ]
    }

# ── 4. Transcription (bonus) ──────────────────────────────────────────────────
def transcribe(mp3_path: str, language: str = None) -> str: # type: ignore
    """Transcribe audio. Optionally force a specific language."""
    audio = load_audio(mp3_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe") if language else None

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            forced_decoder_ids=forced_ids,
            max_new_tokens=448
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# ── 1. Load model & processor ─────────────────────────────────────────────────
MODEL_ID = "DIABOLIC6045/GITAWHISPER-TINY"

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device) # type: ignore
# model.eval()

# ── 5. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ---- DIABOLIC6045/GITAWHISPER-TINY ----
    MP3_FILE = "Atharvaveda_Kanda_10_0001.wav"  
    # audio = whisper.load_audio("ss-audio-may-23.mp3") #sanskrit audio 
                # Top 5 Languages:
                #    cs             0.01%
                #    tl             0.01%
                #    th             0.01%
                #    ta             0.00%
                #    hi             0.00%
    # audio = whisper.load_audio("Atharvaveda_Kanda_10_0001.wav") #sanskrit audio
                # Top 5 Languages:
                #    kn             0.01%
                #    ta             0.01%
                #    en             0.01%
                #    te             0.00%
                #    sa             0.00%
    # audio = whisper.load_audio("bg-chap1.mp3") #background audio
                # Top 5 Languages:
                #    ta             0.05%
                #    en             0.02%
                #    hi             0.01%
                #    sa             0.01%
                #    te             0.01%

    print("🔍 Detecting language...")
    result = detect_language(MP3_FILE)

    print(f"\n✅ Detected Language : {result['detected_language'].upper()}")
    print(f"   Confidence        : {result['confidence']}%")
    print(f"\n📊 Top 5 Languages:")
    for entry in result["top_5"]:
        print(f"   {entry['language']:<12} {entry['probability']:>6.2f}%")

    # print("\n📝 Transcribing...")
    # transcript = transcribe(MP3_FILE, language=result["detected_language"])
    # print(f"\n   {transcript}")