from __future__ import annotations

DEFAULT_ASR_MODELS = {
    "Whisper Small": "openai/whisper-small",
    "Whisper Base": "openai/whisper-base",
    "Wav2Vec2 Turkish": "facebook/wav2vec2-large-xlsr-53-turkish",
}

DEFAULT_SENTIMENT_MODELS = {
    "Savasy Turkish Sentiment": "savasy/bert-base-turkish-sentiment-cased",
    "Turkish DistilBERT Sentiment": "azizbarank/distilbert-base-turkish-cased-sentiment",
}

DEFAULT_ACOUSTIC_MODELS = {
    "DynAnn Speech Emotion": "dynann/emotion-speech-recognition",
    "Wav2Vec2 Emotion": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "MMS Multilingual Encoder": "facebook/mms-1b-all",
}

POSITIVE_THRESHOLD = 0.50
STRESS_THRESHOLD = 0.30
TARGET_SAMPLE_RATE = 16_000
MAX_UPLOAD_MB = 50
