from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.analysis import analyze_audio_file
from src.audio_utils import build_mel_spectrogram_figure, load_audio
from src.config import (
    DEFAULT_ACOUSTIC_MODELS,
    DEFAULT_ASR_MODELS,
    DEFAULT_SENTIMENT_MODELS,
    MAX_UPLOAD_MB,
)

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AcoSemantic-TR",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling and hero area
# -----------------------------------------------------------------------------
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


def _render_hero() -> None:
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

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.header("Model Secimi")
        asr_label = st.selectbox("ASR modeli", list(DEFAULT_ASR_MODELS))
        sentiment_label = st.selectbox("Metin duygu modeli", list(DEFAULT_SENTIMENT_MODELS))
        acoustic_label = st.selectbox("Akustik model", list(DEFAULT_ACOUSTIC_MODELS))
        st.caption("Not: Seçilen modelle tek yollu analiz yapılır; model yükleme hataları kullanıcıya doğrudan bildirilir.")
        st.divider()
        st.markdown(f"**Yukleme limiti:** {MAX_UPLOAD_MB} MB")
        st.markdown("**Hedef ornekleme:** 16 kHz")
        st.divider()

    return asr_label, sentiment_label, acoustic_label

# -----------------------------------------------------------------------------
# Input selection
# -----------------------------------------------------------------------------
def _list_demo_files() -> list[Path]:
    sample_dir = Path("demo_samples")
    return sorted([item for item in sample_dir.iterdir() if item.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]) if sample_dir.exists() else []


def _render_inputs(demo_files: list[Path]):
    input_mode = st.radio("Girdi kaynagi", ["Yuklenen dosya", "Demo klasoru"], horizontal=False)

    selected_demo_file: Path | None = None
    if input_mode == "Demo klasoru":
        if demo_files:
            selected_name = st.selectbox("Demo dosyasi", [item.name for item in demo_files])
            selected_demo_file = Path("demo_samples") / selected_name
            st.caption("Demo dosyasi klasordeki hazir orneklerden secilir.")
        else:
            st.warning("demo_samples klasorunde ses dosyasi bulunamadi.")

    uploaded_file = st.file_uploader(".wav dosyasi yukleyin", type=["wav", "mp3", "flac", "ogg"])
    saved_result_file = st.file_uploader("Kaydedilmis sonuc JSON", type=["json"])

    return input_mode, selected_demo_file, uploaded_file, saved_result_file


# -----------------------------------------------------------------------------
# Result rendering helpers
# -----------------------------------------------------------------------------
def _read_audio_bytes(source_path: Path) -> bytes:
    return source_path.read_bytes()


def _build_result_payload(result, source_name: str, asr_label: str, sentiment_label: str, acoustic_label: str) -> dict[str, object]:
    return {
        "source_name": source_name,
        "models": {
            "asr": {"label": asr_label, "model": DEFAULT_ASR_MODELS[asr_label]},
            "sentiment": {"label": sentiment_label, "model": DEFAULT_SENTIMENT_MODELS[sentiment_label]},
            "acoustic": {"label": acoustic_label, "model": DEFAULT_ACOUSTIC_MODELS[acoustic_label]},
        },
        "result": {
            "transcript": result.transcript,
            "sentiment_label": result.sentiment_label,
            "positivity_score": result.positivity_score,
            "acoustic_label": result.acoustic_label,
            "stress_score": result.stress_score,
            "anomaly_detected": result.anomaly_detected,
            "discordance_score": result.discordance_score,
            "verdict": result.verdict,
            "acoustic_mode": result.acoustic_mode,
            "metadata": result.metadata,
        },
    }


def _render_metric_row(positivity_score: float, stress_score: float, discordance_score: float) -> None:
    metric_cols = st.columns(3)
    metric_cols[0].markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_cols[0].metric("Metin pozitifligi", f"{positivity_score:.2f}")
    metric_cols[0].markdown('</div>', unsafe_allow_html=True)
    metric_cols[1].markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_cols[1].metric("Ses stresi", f"{stress_score:.2f}")
    metric_cols[1].markdown('</div>', unsafe_allow_html=True)
    metric_cols[2].markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_cols[2].metric("Celiski skoru", f"{discordance_score:.2f}")
    metric_cols[2].markdown('</div>', unsafe_allow_html=True)


def _render_saved_result_panel(payload: dict[str, object]) -> None:
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        st.error("Kaydedilmis sonuc okunamadi.")
        return

    st.subheader("Kaydedilmis Sonuc")
    _render_metric_row(
        float(result.get("positivity_score", 0.0)),
        float(result.get("stress_score", 0.0)),
        float(result.get("discordance_score", 0.0)),
    )

    verdict = str(result.get("verdict", "Sonuc bulunamadi."))
    if bool(result.get("anomaly_detected", False)):
        st.error(verdict)
    else:
        st.success(verdict)

    st.markdown(f"**Kaynak:** {payload.get('source_name', 'Bilinmiyor')}")
    models = payload.get("models", {})
    if isinstance(models, dict):
        asr_model = models.get("asr", {})
        sentiment_model = models.get("sentiment", {})
        acoustic_model = models.get("acoustic", {})
        st.markdown(
            f"**Modeller:** ASR `{asr_model.get('label', '-')}` / Sentiment `{sentiment_model.get('label', '-')}` / Akustik `{acoustic_model.get('label', '-')}`"
        )

    st.markdown(f"**Transkript:** {result.get('transcript', 'Metin uretilemedi.') or 'Metin uretilemedi.'}")
    st.markdown(f"**Metin duygu etiketi:** {result.get('sentiment_label', '-')}")
    st.markdown(f"**Akustik etiket:** {result.get('acoustic_label', '-')} ({result.get('acoustic_mode', '-')})")

    with st.expander("Kaydedilen ham ciktilar"):
        st.json(payload)


def _render_result_panel(
    result,
    audio_bytes: bytes,
    audio,
    sample_rate: int,
    source_name: str,
    asr_label: str,
    sentiment_label: str,
    acoustic_label: str,
) -> None:
    payload = _build_result_payload(result, source_name, asr_label, sentiment_label, acoustic_label)

    _render_metric_row(result.positivity_score, result.stress_score, result.discordance_score)

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

            - Metin pozitifliği eşiği: `> 0.50`
            - Ses stresi eşiği: `> 0.30`
            - Mevcut durum: `{result.positivity_score:.2f}` / `{result.stress_score:.2f}`
            - Sonuc: `{'Anomali' if result.anomaly_detected else 'Normal akış'}`
            """
        )
        st.download_button(
            "Sonucu JSON olarak indir",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name=f"{Path(source_name).stem}_result.json",
            mime="application/json",
        )
        st.audio(audio_bytes)
    with spectrogram_col:
        st.subheader("Mel-Spektrogram")
        figure = build_mel_spectrogram_figure(audio, sample_rate)
        st.pyplot(figure)
        plt.close(figure)

    with st.expander("Ham ciktilar"):
        st.json(payload)


# -----------------------------------------------------------------------------
# Main interaction flow
# -----------------------------------------------------------------------------
analysis_source: Path | None = None
analysis_bytes: bytes | None = None

def _analyze_and_render(
    source_path: Path,
    source_name: str,
    audio_bytes: bytes,
    asr_label: str,
    sentiment_label: str,
    acoustic_label: str,
) -> None:
    with st.spinner("Ses, metin ve duygu hatlari isleniyor..."):
        result = analyze_audio_file(
            source_path,
            DEFAULT_ASR_MODELS[asr_label],
            DEFAULT_SENTIMENT_MODELS[sentiment_label],
            DEFAULT_ACOUSTIC_MODELS[acoustic_label],
        )
        audio, sample_rate = load_audio(source_path)

    _render_result_panel(
        result,
        audio_bytes,
        audio,
        sample_rate,
        source_name,
        asr_label,
        sentiment_label,
        acoustic_label,
    )


def _handle_uploaded_file(uploaded_file, asr_label: str, sentiment_label: str, acoustic_label: str) -> None:
    uploaded_bytes = uploaded_file.getvalue()
    if len(uploaded_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("Dosya boyutu limitin ustunde.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_bytes)

        if st.button("Analizi calistir", type="primary"):
            _analyze_and_render(temp_path, uploaded_file.name, uploaded_bytes, asr_label, sentiment_label, acoustic_label)


def _handle_demo_file(selected_demo_file: Path, asr_label: str, sentiment_label: str, acoustic_label: str) -> None:
    audio_bytes = _read_audio_bytes(selected_demo_file)
    if st.button("Secili demo ile calistir", type="primary"):
        _analyze_and_render(selected_demo_file, selected_demo_file.name, audio_bytes, asr_label, sentiment_label, acoustic_label)


def _handle_saved_result_file(saved_result_file) -> None:
    try:
        saved_payload = json.loads(saved_result_file.read().decode("utf-8"))
        st.markdown("---")
        _render_saved_result_panel(saved_payload)
    except Exception:
        st.error("Kaydedilmis JSON dosyasi okunamadi.")


_render_hero()
asr_label, sentiment_label, acoustic_label = _render_sidebar()
demo_files = _list_demo_files()
input_mode, selected_demo_file, uploaded_file, saved_result_file = _render_inputs(demo_files)

if input_mode == "Yuklenen dosya":
    if uploaded_file is not None:
        _handle_uploaded_file(uploaded_file, asr_label, sentiment_label, acoustic_label)
    else:
        st.info("Bir ses dosyasi yukleyin veya demo klasorunden dosya secin.")
elif selected_demo_file is not None:
    _handle_demo_file(selected_demo_file, asr_label, sentiment_label, acoustic_label)

if not uploaded_file and not selected_demo_file:
    st.info("Bir ses dosyasi yukleyin veya demo_samples klasorundeki hazir orneklerden birini secin.")

if saved_result_file is not None:
    _handle_saved_result_file(saved_result_file)

if demo_files:
    st.markdown("**Hazir demo dosyalari:**")
    for item in demo_files:
        st.markdown(f"- {item.name}")

st.markdown("---")
st.caption("Bu demo, ironi, kinaye ve yapay sakinlik gibi duygu maskeleri icin Turkce odakli bir referans iskeletidir.")
