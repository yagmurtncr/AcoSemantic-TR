# Sistem Mimarisi

```mermaid
flowchart LR
    U[Ses Girdisi] --> A[On Isleme]
    A --> B[ASR Katmani]
    A --> C[Akustik Katmani]
    B --> D[Turkce Transkript]
    D --> E[Sentiment BERT]
    E --> F[Pozitiflik Skoru]
    C --> G[Stres Skoru]
    F --> H[Karar Mekanizmasi]
    G --> H
    H --> I[Anomali / Normal]
```

## Bilesenler

- ASR: Whisper Small ana model, Whisper Base ve Wav2Vec2 Turkish yedek
- Akustik: Wav2Vec2 emotion modelleri ana yol, MMS encoder heuristik yedek
- NLP: Turkce BERT sentiment sinifi
- UI: Streamlit + mel-spektrogram gorunumu

## Sunum Mesaji

Sistem, yalnizca metni degil, tonlamayi da analiz ederek duygu maskelemesini tespit eder. Bu nedenle klasik sentiment siniflayicidan daha zengin bir sinyal birlesimi sunar.
