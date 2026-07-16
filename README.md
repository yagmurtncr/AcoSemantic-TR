# AcoSemantic-TR

<p>
  <img src="https://github.com/yagmurtncr/AcoSemantic-TR/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

> **Acoustic–semantic emotion analysis for Turkish speech.** Detects emotion mismatches and
> anomalies by jointly evaluating textual meaning and acoustic features (tone, stress), exposing a
> production-ready analysis layer with reusable JSON output (FastAPI + Streamlit). *Docs in Turkish.*

## AcoSemantic-TR — Kısa Tanım

AcoSemantic-TR, Türkçe konuşmalarda metin anlamı ile sesin akustik özelliklerini (ton, stres) birlikte değerlendirerek duygu çelişkilerini ve anomalileri tespit eden bir analiz servisidir.

Hedef: Uçtan uca çalışan, tekrar kullanılabilir JSON çıktısı veren ve üretime alınmaya uygun bir analiz katmanı sunmak.

**Yeni (v1.1):** Akustik taraf artık yalnızca tek bir modele bağlı değil — **modelden bağımsız,
sıfırdan yazılmış prozodik özellik çıkarımı** (pitch/F0, enerji, konuşma hızı, duraklamalar) eklendi.
Bu sayede akustik analiz gerçekten "ton/stres" ölçer ve **akustik model yoksa bile** (prozodi fallback)
sistem çalışmaya devam eder. Ayrıca eşikleri **gerçek veriyle kalibre eden** bir değerlendirme modülü var.

## Mimari (Architecture)

```mermaid
flowchart LR
    AUDIO["🔊 Speech audio"] --> ASR["ASR → text"]
    AUDIO --> ACO["Acoustic features<br/>(pitch, tone, stress)"]

    ASR --> SEM["Semantic emotion<br/>(text meaning)"]
    ACO --> AEMO["Acoustic emotion<br/>(voice)"]

    SEM --> CMP{"Compare<br/>semantic vs acoustic"}
    AEMO --> CMP
    CMP -->|"mismatch / anomaly"| FLAG["⚠️ Emotion conflict flagged"]
    CMP -->|"consistent"| OK["Consistent"]

    FLAG --> JSON["Structured JSON report"]
    OK --> JSON
    JSON --> API["FastAPI / Streamlit"]
```

## Hızlı Başlangıç

1. Sanal ortam oluşturun ve etkinleştirin (ör. venv).
2. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

3. Hugging Face token'ınızı ayarlayın (bakınız Token Yönetimi).

## Token Yönetimi

Model erişimi için `HF_TOKEN` veya `HUGGINGFACE_HUB_TOKEN` gereklidir. Bu token'ı repoya eklemeyin; geçici olarak PowerShell'de ayarlayabilir veya yerel, .gitignored `.streamlit/secrets.toml` içine koyabilirsiniz:

```toml
HF_TOKEN = "your_hugging_face_token"
HUGGINGFACE_HUB_TOKEN = "your_hugging_face_token"
```

PowerShell örneği:

```powershell
$env:HF_TOKEN = "your_hugging_face_token"
$env:HUGGINGFACE_HUB_TOKEN = $env:HF_TOKEN
streamlit run app.py
```

## Nasıl Çalıştırılır

API başlatmak için:

```bash
uvicorn api:app --reload
```

Sağlık kontrolü:

```bash
curl http://127.0.0.1:8000/health
```

Dosya ile analiz örneği (curl):

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@demo_samples/ornek.wav" \
  -F "asr_model=Whisper Small" \
  -F "sentiment_model=Savasy Turkish Sentiment" \
  -F "acoustic_model=DynAnn Speech Emotion"
```

Streamlit UI çalıştırmak için:

```bash
streamlit run app.py
```

## 🧪 Test & Kod Kalitesi

Karar mantığı (`src/decision.py`) ses I/O ve modellerden ayrılmıştır; bu sayede saf,
hızlı ve **ağır bağımlılık gerektirmeden** test edilebilir.

```bash
ruff check .     # lint (CI'da zorunlu)
pytest -q        # birim testleri (~0.1 sn, torch/librosa gerekmez)
```

- **CI** — GitHub Actions her push/PR'da `ruff` + `pytest` çalıştırır
- **Testler** karar motorunun tüm dallarını (maskeleme anomalisi, sınır durumları,
  ayarlanabilir eşikler) ve config bütünlüğünü kapsar

---

## 🎚️ Prozodik Analiz (model-free) & Değerlendirme

**`src/prosody.py`** — saf NumPy DSP ile ham dalga formundan yorumlanabilir prozodik öznitelikler:

| Öznitelik | Ne ölçer |
|-----------|----------|
| `pitch_mean_hz` / `pitch_std_hz` | Otokorelasyon-tabanlı F0 ve **pitch değişkenliği** (uyarılma/stres) |
| `rms_mean` / `rms_std` | Ses şiddeti ve değişkenliği |
| `zcr_mean` | Sıfır-geçiş oranı (konuşma hızı/gürültü vekili) |
| `pause_ratio` / `voiced_ratio` | Sessizlik/duraklama oranı |
| `stress_index` (0–1) | Yukarıdakilerden türetilen **birleşik prozodik stres skoru** |

Bunlar her analize eklenir; akustik model yüklenemezse `stress_index` **fallback** olarak kullanılır.

**`src/evaluation.py`** — README'nin "eşikleri gerçek veriyle kalibre et" hedefini karşılar:
`precision / recall / F1 / accuracy` metrikleri + karar eşiklerini **F1'e göre grid-search ile kalibre eden**
`calibrate_thresholds(...)`. RAVDESS gibi etiketli verilerle çalışır.

```python
from src.evaluation import calibrate_thresholds
# samples: (positivity, stress, is_discordant) ...
best = calibrate_thresholds(samples)
print(best.positive_threshold, best.stress_threshold, best.metrics["f1"])
```

## Klasör Yapısı (Önemli Dosyalar)

- `api.py` — FastAPI endpoint'leri
- `app.py` — Streamlit arayüzü
- `src/analysis.py` — Uçtan uca analiz akışı (prozodi + fallback dahil)
- `src/decision.py` — Duygu-çelişkisi karar motoru (saf, test edilebilir)
- `src/prosody.py` — Model-free prozodik özellik çıkarımı (NumPy DSP)
- `src/evaluation.py` — Metrikler + eşik kalibrasyonu
- `src/models.py` — Model yükleme ve çıkarım
- `demo_samples/` — Örnek sesler ve demo çıktıları

## Varsayılan Modeller ve Eşikler

- ASR: `openai/whisper-small` (varsayılan)
- Sentiment: `savasy/bert-base-turkish-sentiment-cased` (varsayılan)
- Akustik: `dynann/emotion-speech-recognition` (varsayılan)
- Karar eşikleri (varsayılan, veriyle doğrulanmalı): Metin_Pozitiflik > 0.50, Ses_Stres > 0.30

## Docker (Hızlı)

Üretim benzeri tek servis çalıştırma için:

```bash
docker build -t acosemantic-tr-api .
docker run --rm -p 8000:8000 --env-file .env acosemantic-tr-api
```

## Notlar ve İpuçları

- Model yükleme başarısız olursa uygulama doğrudan hata verir — bu proje model tabanlı çalışmayı zorunlu kılar.
- `demo_samples/` klasörüne test sesleri konulursa arayüz otomatik listeler.

## Geliştirme ve Yayına Hazırlık

1. ✅ Eşik kalibrasyonu artık `src/evaluation.py` ile veri-güdümlü yapılabilir (F1 grid-search).
2. Smoke testleri ve temel birim testlerini çalıştırın.
3. Kod formatlayıcı ve linter (Black, isort, ruff/flake8) çalıştırın.
4. Üretim `.env` ve secret yönetimini doğrulayın.

---

İhtiyorsanız README'yi daha kısa bir kullanıcı rehberi veya dağıtım talimatına da indirebilirim.
