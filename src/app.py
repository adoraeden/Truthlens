
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="DeepfakeLab",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 DeepfakeLab")
st.subheader("AI-Generated Image Detection")
st.write(
    "Upload an image to prepare it for authenticity analysis. "
    "Model predictions will appear once the detector is integrated."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded image", use_container_width=True)
        st.write(f"Image dimensions: {image.width} × {image.height}")

        st.info(
            "Image received successfully. The detection model "
            "has not been connected yet, so no authenticity result is available."
        )
    except Exception:
        st.error("Unable to read this image. Please try another file.")

st.divider()
st.caption(
    "Research prototype • Image detection first • "
    "A prediction is not proof of authenticity."
)
