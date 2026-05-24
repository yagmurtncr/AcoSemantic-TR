# AcoSemantic-TR — İlerleme Özeti

Aşağıda proje kapsamında şimdiye kadar yaptığım işler ve takip edilecek kısa adımlar yer alıyor.

## Yapılanlar (özet)
- Proje iskeleti ve ana modüller korunarak demo pipeline hazırlandı (ASR, semantik duygu, akustik stres, karar motoru).
- `scripts/download_extract_ravdess.py` ile RAVDESS için indirme-senaryosu eklendi; sonrasında `demo_samples/` altında üç örnek WAV kaydı mevcut.
- HF model erişim hatalarına karşı `src/models.py` içinde fallback heuristikleri eklendi (metin için anahtar kelime temelli ve akustik için prosodi tabanlı fallbacks).
- `scripts/run_demo_analysis.py` ile demo klasöründeki WAV'ları analiz edip `demo_samples/demo_results.json` yazdırdım.
- Streamlit arayüzü (`app.py`) başlatıldı; onboarding e-posta girildi ve uygulama çalışır durumda.
- Kullanıcının talebi üzerine Türkçe veri araması yapıldı ve açık kaynak Türkçe duygusal konuşma seti `TurEV-DB` bulundu ve `external_datasets/` içine klonlandı.
- `TurEV-DB` içinden seçilen birkaç örnek `demo_samples/` içine kopyalandı: `tur_ev_happy.wav`, `tur_ev_calm.wav`, `tur_ev_angry.wav`.
- Toplu çalışma için yardımcı script eklendi: `scripts/select_and_copy_tur_ev.py` (TurEV'den rastgele 20 WAV seçip `demo_samples/` kopyalamak için).

## Çalıştırılan temel komutlar
- `.
.venv\Scripts\python.exe scripts/run_demo_analysis.py` — demo analizini çalıştırır ve `demo_results.json` üretir.
- `.
.venv\Scripts\python.exe -m streamlit run app.py` — Streamlit UI'yi başlatır.
- `git clone https://github.com/Xeonen/TurEV-DB.git external_datasets/TurEV-DB` — TurEV-DB depo klonlandı.
- `.
.venv\Scripts\python.exe scripts\select_and_copy_tur_ev.py` — (yardımcı) 20 rastgele TurEV WAV'ını `demo_samples/` içine kopyalar.

## Mevcut durum
- `demo_samples/demo_results.json` güncel ve içinde hem önceki kontrollü demo hem de TurEV örnek analizleri bulunuyor.
- Streamlit arayüzü çalışıyor (localhost:8501 tipik port).
- Akustik model artık `dynann/emotion-speech-recognition` ile model tabanlı çalışıyor; `acoustic_mode` alanı çoğu örnekte `classifier` olarak dönüyor.
- Eşik optimizasyonu sonucu en iyi çalışan karar eşikleri şu an için `POSITIVE_THRESHOLD = 0.50` ve `STRESS_THRESHOLD = 0.30`.
- Bu eşiklerle demo verisinde sadece iki kontrollü örnek anomali veriyor: `calm_words_high_stress.wav` ve `positive_words_angry_tone.wav`.

## Sonraki adımlar (önerilen öncelik sıralı)
1. `scripts/select_and_copy_tur_ev.py` ile TurEV'den 20 örnek kopyalanıp tam batch analizini çalıştırmak (in-progress).
2. Streamlit arayüzünü sonuçlarla yeniden entegre etmek — yeni `demo_results.json` yüklenip görselleştirme sağlanacak.
3. HF Hub erişimi için `HF_TOKEN` (kullanıcı temin ederse) ile modellerin doğruluğu ve hızını artırmak.
4. Eğer gerekiyorsa TurEV-DB dışında ilave Türkçe duygu setleri araştırılacak veya kayıt/etiketleme süreci başlatılacak.
5. Değerlendirme: semantik model doğruluğu (Türkçe BERT) ve akustik model/heuristik performansı karşılaştırılıp karar motoru eşikleri (`POSITIVE_THRESHOLD`, `STRESS_THRESHOLD`) gerekirse ayarlanacak.

## Son karar
- Akustik taraf için başarılı model: `dynann/emotion-speech-recognition`.
- Karar kuralı için şu anki en iyi eşikler: `POSITIVE_THRESHOLD = 0.50`, `STRESS_THRESHOLD = 0.30`.
- Bu kombinasyon, mevcut demo veri setinde yanlış pozitif üretmeden iki hedef anomaliyi yakalıyor.

## Notlar / Gereksinimler
- Bazı HF modelleri ağ ve kimlik doğrulama (token) gerektirebilir; demo sırasında fallback mekanizmaları devreye girer.
- Lisans: `TurEV-DB` CC-BY-NC-ND gibi kısıtlar içerebilir; kullanım amacınıza göre lisans inceleyin.

İsterseniz bu dosyayı genişletip commit mesajı, daha ayrıntılı çalışma adımları veya otomatikleştirilmiş test talimatları ekleyebilirim.
