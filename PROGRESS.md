 # AcoSemantic-TR — Proje Özeti

Aşağıdaki özet projeyi kısa ve profesyonel bir şekilde sunar: ne yaptığı, proje yapısı, tamamlanan işler ve bir sonraki adımlar.

## Proje Amacı
AcoSemantic-TR, Türkçe konuşmalarda sözcük anlamı (metin) ile sesin akustik özelliklerini (ton, stres) birlikte değerlendirerek duygu çelişkilerini ve anomalileri tespit etmeye odaklanan bir üründür.

Hedef, yalnızca bir demo sunmak değil; uçtan uca çalışan, sonuçları insan okunabilir biçimde döken ve üretime alınmaya uygun bir analiz servisi geliştirmektir.

Analiz akışı kısaca:
- Girdi: ses dosyası
- Yol 1: ASR → metin transkripsiyonu → BERT tabanlı sentiment (pozitiflik)
- Yol 2: Akustik model → ton/stres skoru
- Karar: İki skor birleştirilir; ortak yükseklik durumlarında anomali/çelişki raporlanır.

## Kod ve Dizin Yapısı
- `api.py`: FastAPI tabanlı HTTP endpoint'leri
- `app.py`: Streamlit kullanıcı arayüzü
- `src/analysis.py`: Analiz ve karar mekanizması
- `src/models.py`: Model yükleme ve çıkarım (ASR, sentiment, akustik)
- `src/audio_utils.py`: Ses I/O ve spektrogram görselleştirme
- `src/config.py`: Varsayılan modeller ve eşik değerleri
- `demo_samples/`: Örnek sesler ve demo çıktıları
- `README.md`: Kurulum ve kullanım rehberi

## Kısa Özet — Neler Yapıldı
- Ses yükleme, demo seçimi ve toplu analiz akışı çalışır durumda.
- ASR, sentiment ve akustik analiz ile karar motoru entegre edildi.
- Hugging Face token desteği eklendi (çevresel değişken veya `.streamlit/secrets.toml`).
- Model tabanlı (heuristic-free) analiz akışı kuruldu.
- FastAPI ile `/health` ve `/analyze` endpointleri eklendi ve smoke-test ile doğrulandı.
- Streamlit arayüzünde sadeleştirilmiş sonuç ekranı, JSON indir/göster desteği ve kaydet/yeniden açma eklendi.
- Batch analiz ve özet üretme scriptleri eklendi.
- Docker için paketlenebilir yapı, `.env.example` ve `.dockerignore` eklendi.

## Denenen Modeller ve Varsayılan Eşikler
- ASR: openai/whisper-small (varsayılan), openai/whisper-base, facebook/wav2vec2-large-xlsr-53-turkish
- Sentiment: savasy/bert-base-turkish-sentiment-cased (varsayılan), azizbarank/distilbert-base-turkish-cased-sentiment
- Akustik: dynann/emotion-speech-recognition (varsayılan), ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition, facebook/mms-1b-all
- Karar eşikleri (geçici, veriyle doğrulanmalı): Metin_Pozitiflik > 0.50, Ses_Stres > 0.30

## İyileştirme Fırsatları
- Eşikleri gerçek dünya verisiyle yeniden kalibre etmek.
- Daha büyük ve temiz Türkçe veri setleriyle karşılaştırmalı denemeler yapmak.
- Hata ve uyarı mesajlarını kullanıcı-odaklı hale getirmek.
- CI/CD ve gizli anahtar yönetimini (secrets) üretim politikalarına göre kurmak.

## Güncel Yapılacaklar (Önceliklendirilmiş)
1. README ve PROGRESS içeriklerini profesyonel, son kullanıcı dostu hâle getirmek.  
2. `app.py` dosyasını küçük, tek-sorumluluklu fonksiyonlara bölmek ve yeniden test etmek.  
3. Smoke ve temel birim testlerini çalıştırmak / eklemek.  
4. Kod formatlama ve lint çalıştırmak (Black, isort, flake8/ruff).  
5. Son onay: üretim `.env` ve secret yönetimini doğrulamak ve deploy öncesi adımları tamamlamak.

## Kısa Durum
Çekirdek analiz doğrulandı ve API ile UI aynı analizi paylaşıyor. Proje artık ürünleştirme aşamasında; bir sonraki hedef kod kalitesini yükseltmek, eşik doğrulamalarını tamamlamak ve dağıtıma hazır hâle getirmektir.
