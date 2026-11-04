
import tempfile
from pathlib import Path

import streamlit as st

from image_detector import ImageDetector
from audio_detector import AudioDetector
from video_detector import VideoDetector


st.set_page_config(
    page_title="TruthLens | Media Authenticity",
    page_icon="🔍",
    layout="wide",
)

AUDIO_UNAVAILABLE_REASON = None

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }
    .hero {
        padding: 2rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }
    .muted {
        color: #888;
        font-size: .9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading image detection model...")
def get_image_detector():
    return ImageDetector()


@st.cache_resource(show_spinner="Loading audio detection model...")
def get_audio_detector():
    global AUDIO_UNAVAILABLE_REASON
    try:
        AUDIO_UNAVAILABLE_REASON = None
        return AudioDetector()
    except RuntimeError as error:
        AUDIO_UNAVAILABLE_REASON = str(error)
        return None


@st.cache_resource(show_spinner="Preparing video analysis...")
def get_video_detector():
    audio_detector = get_audio_detector()
    return VideoDetector(
        get_image_detector(),
        audio_detector,
        audio_unavailable_reason=AUDIO_UNAVAILABLE_REASON,
    )


def save_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:
        temp.write(uploaded_file.getbuffer())
        return Path(temp.name)


def show_result(result):
    verdict = result["verdict"]
    score = result["score"]

    if verdict == "AI-generated":
        st.error("### AI-generated indicators detected")
    else:
        st.success("### Likely real according to this model")

    st.metric("Uncalibrated model score", f"{score * 100:.2f}%")

    st.progress(max(0.0, min(1.0, float(score))))

    st.caption(
        "This is an uncalibrated model score, not a calibrated probability "
        "that the media is fake."
    )


def show_image_result(result):
    show_result(result)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model logit", f"{result['logit']:.3f}")
    col2.metric("Raw-logit decision cutoff", f"≥ {result['decision_threshold']:g}")
    col3.write(f"Image dimensions: {result['image_size'][0]} × {result['image_size'][1]}")


def show_audio_result(result):
    show_result(result)
    st.info(result["note"])
    st.caption(
        f"Duration analysed: {result['duration_analyzed_seconds']} seconds"
    )


def show_video_result(result):
    st.subheader("Video frame analysis")

    st.metric("Frames analysed", result["frames_analyzed"])
    st.metric("Frames flagged as AI", result["ai_flagged_frames"])
    st.metric(
        "Average frame fake score",
        f"{result['average_frame_score'] * 100:.2f}%",
    )

    for index, frame in enumerate(result["frame_results"], start=1):
        with st.expander(
            f"Frame {index} — {frame['timestamp_seconds']} seconds"
        ):
            st.write(f"Verdict: **{frame['verdict']}**")
            st.write(f"Model fake score: {frame['score'] * 100:.2f}%")
            st.write(f"Model logit: {frame['logit']:.3f}")

    st.subheader("Video audio analysis")

    if result["audio_analysis"] is not None:
        show_audio_result(result["audio_analysis"])
    elif result.get("audio_status") == "audio_detector_unavailable":
        st.warning(result.get("audio_status_note", "The audio detector was not initialized; no audio conclusion was made."))
    else:
        st.warning(result.get("audio_status_note", "No usable audio track was detected in this video."))

    st.info(result["interpretation"])


st.markdown("""
<div class="hero">
    <h1>🔍 TruthLens</h1>
    <h3>AI-powered media authenticity analysis</h3>
    <p>
        Analyse images, audio recordings and videos for indicators
        of AI-generated or manipulated media.
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "TruthLens provides automated screening, not proof of authenticity. "
    "A false positive or missed synthetic sample is possible."
)

image_tab, audio_tab, video_tab = st.tabs(
    ["🖼️ Image Analysis", "🎙️ Audio Analysis", "🎬 Video Analysis"]
)


with image_tab:
    st.header("Image authenticity")
    st.write("Upload an image for AI-generation screening.")

    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_upload",
    )

    if image_file:
        st.image(image_file, caption=image_file.name, use_container_width=True)

        if st.button("Analyse image", type="primary"):
            temp_path = save_upload(image_file)

            try:
                with st.spinner("Analysing image..."):
                    result = get_image_detector().predict(temp_path)

                show_image_result(result)

            except Exception as error:
                st.error(f"Image analysis failed: {error}")

            finally:
                temp_path.unlink(missing_ok=True)


with audio_tab:
    st.header("Audio authenticity")
    st.write("Upload an audio recording for synthetic speech screening.")

    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        key="audio_upload",
    )

    if audio_file:
        st.audio(audio_file)

        if st.button("Analyse audio", type="primary"):
            temp_path = save_upload(audio_file)

            try:
                audio_detector = get_audio_detector()
                if audio_detector is None:
                    st.warning(AUDIO_UNAVAILABLE_REASON or "Audio detector is unavailable.")
                else:
                    with st.spinner("Analysing audio... This may take a moment."):
                        result = audio_detector.predict(temp_path)

                    show_audio_result(result)

            except Exception as error:
                st.error(f"Audio analysis failed: {error}")

            finally:
                temp_path.unlink(missing_ok=True)


with video_tab:
    st.header("Video authenticity")
    st.write(
        "Analyse sampled video frames and the audio track separately."
    )

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "mkv", "avi"],
        key="video_upload",
    )

    if video_file:
        st.video(video_file)

        if st.button("Analyse video", type="primary"):
            temp_path = save_upload(video_file)

            try:
                with st.spinner(
                    "Analysing video frames and audio. Please wait..."
                ):
                    result = get_video_detector().predict(temp_path)

                st.success("Video analysis completed.")
                st.metric(
                    "Video duration",
                    f"{result['duration_seconds']:.2f} seconds",
                )
                show_video_result(result)

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                temp_path.unlink(missing_ok=True)


st.divider()

st.markdown("""
### About TruthLens

TruthLens uses separate machine-learning detectors for image and audio
analysis. Video analysis samples frames and evaluates the extracted
audio independently.

**Important limitations**
- Model scores are not calibrated probabilities.
- Real media can be incorrectly flagged.
- Synthetic media can evade detection.
- Audio analysis currently examines a four-second centre crop.
- Video analysis samples up to eight frames, not every frame.
- A result is an indicator for further review, not a final authenticity judgment.
""")

st.caption("TruthLens • Media authenticity screening • Hackathon prototype")
