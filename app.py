from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.analysis import analyze_audio_file
from src.audio_utils import build_mel_spectrogram_figure, load_audio
from src.config import DEFAULT_ACOUSTIC_MODELS, DEFAULT_ASR_MODELS, DEFAULT_SENTIMENT_MODELS, MAX_UPLOAD_MB


st.set_page_config(
    page_title="AcoSemantic-TR",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(12, 74, 110, 0.28), transparent 28%),
                radial-gradient(circle at top right, rgba(0, 0, 0, 0.22), transparent 24%),
                linear-gradient(180deg, #0b1220 0%, #10192d 42%, #0d1322 100%);
            color: #e5eef8;
        }
        .hero {
            padding: 2rem 2rem 1.2rem 2rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 24px;
            background: rgba(8, 15, 30, 0.72);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin-top: 0.75rem;
            color: #b6c5da;
            line-height: 1.6;
        }
        .card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.72);
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(125, 211, 252, 0.16);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(8, 15, 30, 0.92));
        }
        .muted {
            color: #9fb0c7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AcoSemantic-TR</h1>
        <p>
            Turkce konusmada semantik anlam ile akustik tonu ayni anda inceleyen
            anomali tespit demosu. Ses tonu ile kelime duygusu celiskirse sistem uyarir.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Secimi")
    asr_label = st.selectbox("ASR modeli", list(DEFAULT_ASR_MODELS))
    sentiment_label = st.selectbox("Metin duygu modeli", list(DEFAULT_SENTIMENT_MODELS))
    acoustic_label = st.selectbox("Akustik model", list(DEFAULT_ACOUSTIC_MODELS))
    st.caption("Not: MMS modeli bir encoder olarak yedeklenmistir; dogrudan duygu sinifi vermediginde heuristik fallback kullanilir.")
    st.divider()
    st.markdown(f"**Yukleme limiti:** {MAX_UPLOAD_MB} MB")
    st.markdown("**Hedef ornekleme:** 16 kHz")
    st.divider()

sample_dir = Path("demo_samples")
demo_files = sorted(
    [item for item in sample_dir.iterdir() if item.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
) if sample_dir.exists() else []

input_mode = st.radio("Girdi kaynagi", ["Yuklenen dosya", "Demo klasoru"], horizontal=False)

selected_demo_file: Path | None = None
if input_mode == "Demo klasoru":
    if demo_files:
        selected_name = st.selectbox("Demo dosyasi", [item.name for item in demo_files])
        selected_demo_file = sample_dir / selected_name
        st.caption("Demo dosyasi klasordeki hazir orneklerden secilir.")
    else:
        st.warning("demo_samples klasorunde ses dosyasi bulunamadi.")

uploaded_file = st.file_uploader(".wav dosyasi yukleyin", type=["wav", "mp3", "flac", "ogg"])

def _read_audio_bytes(source_path: Path) -> bytes:
    return source_path.read_bytes()


def _render_result_panel(
    result,
    audio_bytes: bytes,
    audio,
    sample_rate: int,
    source_name: str,
) -> None:
    metric_cols = st.columns(3)
    metric_cols[0].markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_cols[0].metric("Metin pozitifligi", f"{result.positivity_score:.2f}")
    metric_cols[0].markdown('</div>', unsafe_allow_html=True)
    metric_cols[1].markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_cols[1].metric("Ses stresi", f"{result.stress_score:.2f}")
    metric_cols[1].markdown('</div>', unsafe_allow_html=True)
    metric_cols[2].markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_cols[2].metric("Celiski skoru", f"{result.discordance_score:.2f}")
    metric_cols[2].markdown('</div>', unsafe_allow_html=True)

    if result.anomaly_detected:
        st.error(result.verdict)
    else:
        st.success(result.verdict)

    summary_col, spectrogram_col = st.columns([1.15, 1])
    with summary_col:
        st.subheader("Karar Ozeti")
        st.markdown(f"**Kaynak:** {source_name}")
        st.markdown(f"**Transkript:** {result.transcript or 'Metin uretilemedi.'}")
        st.markdown(f"**Metin duygu etiketi:** {result.sentiment_label}")
        st.markdown(f"**Akustik etiket:** {result.acoustic_label} ({result.acoustic_mode})")
        st.markdown(
            f"""
            **Karar kuralı**

            - Metin pozitifligi eşiği: `> 0.70`
            - Ses stresi eşiği: `> 0.60`
            - Mevcut durum: `{result.positivity_score:.2f}` / `{result.stress_score:.2f}`
            - Sonuc: `{'Anomali' if result.anomaly_detected else 'Normal akış'}`
            """
        )
        st.audio(audio_bytes)
    with spectrogram_col:
        st.subheader("Mel-Spektrogram")
        figure = build_mel_spectrogram_figure(audio, sample_rate)
        st.pyplot(figure)
        plt.close(figure)

    with st.expander("Ham ciktilar"):
        st.json(result.metadata)


analysis_source: Path | None = None
analysis_bytes: bytes | None = None

if input_mode == "Yuklenen dosya":
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        if len(uploaded_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error("Dosya boyutu limitin ustunde.")
            st.stop()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / uploaded_file.name
            temp_path.write_bytes(uploaded_bytes)
            analysis_source = temp_path
            analysis_bytes = uploaded_bytes

            if st.button("Analizi calistir", type="primary"):
                with st.spinner("Ses, metin ve duygu hatlari isleniyor..."):
                    result = analyze_audio_file(
                        temp_path,
                        DEFAULT_ASR_MODELS[asr_label],
                        DEFAULT_SENTIMENT_MODELS[sentiment_label],
                        DEFAULT_ACOUSTIC_MODELS[acoustic_label],
                    )
                    audio, sample_rate = load_audio(temp_path)

                _render_result_panel(result, uploaded_bytes, audio, sample_rate, uploaded_file.name)
    else:
        st.info("Bir ses dosyasi yukleyin veya demo klasorunden dosya secin.")
elif selected_demo_file is not None:
    analysis_source = selected_demo_file
    analysis_bytes = _read_audio_bytes(selected_demo_file)

    if st.button("Secili demo ile calistir", type="primary"):
        with st.spinner("Demo dosyasi isleniyor..."):
            result = analyze_audio_file(
                selected_demo_file,
                DEFAULT_ASR_MODELS[asr_label],
                DEFAULT_SENTIMENT_MODELS[sentiment_label],
                DEFAULT_ACOUSTIC_MODELS[acoustic_label],
            )
            audio, sample_rate = load_audio(selected_demo_file)

        _render_result_panel(result, analysis_bytes, audio, sample_rate, selected_demo_file.name)

if not uploaded_file and not selected_demo_file:
    st.info("Bir ses dosyasi yukleyin veya demo_samples klasorundeki hazir orneklerden birini secin.")

if demo_files:
    st.markdown("**Hazir demo dosyalari:**")
    for item in demo_files:
        st.markdown(f"- {item.name}")

st.markdown("---")
st.caption("Bu demo, ironi, kinaye ve yapay sakinlik gibi duygu maskeleri icin Turkce odakli bir referans iskeletidir.")
