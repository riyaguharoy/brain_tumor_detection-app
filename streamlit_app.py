"""
Brain Tumor Detection — Streamlit Web App
------------------------------------------
Loads a trained Keras CNN (input 64x64x3, single sigmoid output) and lets
a user upload an MRI image to get a Tumor / Healthy prediction.

Setup:
1. In your training notebook, after model.fit(...), add:
       model.save("brain_tumor_model.h5")
2. Put brain_tumor_model.h5 in the same folder as this script.
3. pip install streamlit tensorflow pillow numpy
4. Run:  streamlit run streamlit_app.py
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "brain_tumor_model.h5"
IMG_SIZE = (64, 64)             # confirmed via model.input_shape -> (None, 64, 64, 3)
# Model's final layer is Dense(1) with sigmoid -> single probability, not a 2-class softmax.
# Confirmed via train_generator.class_indices: {'Brain Tumor': 0, 'Healthy': 1}
CLASS_IF_HIGH = "Healthy"      # sigmoid output near 1
CLASS_IF_LOW = "Brain Tumor"   # sigmoid output near 0


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize + normalize exactly like the training pipeline did."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # same 0-1 normalization as training
    arr = np.expand_dims(arr, axis=0)                # add batch dimension -> (1, 64, 64, 3)
    return arr


def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
    st.title("🧠 Brain Tumor Detection")
    st.caption(
        "Upload a brain MRI scan and the CNN model will classify it as "
        "**Healthy** or **Tumor**."
    )
    st.warning(
        "⚠️ Research / educational demo only. This is **not** a medical "
        "diagnostic tool and must not be used to make real clinical decisions. "
        "Always consult a qualified radiologist or physician.",
        icon="⚠️",
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_container_width=True)

        with st.spinner("Analyzing..."):
            input_tensor = preprocess_image(image)
            raw_output = model.predict(input_tensor)[0][0]  # single sigmoid probability

        # sigmoid near 1 -> CLASS_IF_HIGH, sigmoid near 0 -> CLASS_IF_LOW
        if raw_output >= 0.5:
            predicted_label = CLASS_IF_HIGH
            confidence = float(raw_output) * 100
        else:
            predicted_label = CLASS_IF_LOW
            confidence = float(1 - raw_output) * 100

        st.subheader("Result")
        if predicted_label == "Brain Tumor":
            st.error(f"🔴 Prediction: **{predicted_label}** ({confidence:.1f}% confidence)")
        else:
            st.success(f"🟢 Prediction: **{predicted_label}** ({confidence:.1f}% confidence)")

        st.write(f"Raw model output (probability of '{CLASS_IF_HIGH}'): {raw_output:.4f}")
        st.progress(float(raw_output))


if __name__ == "__main__":
    main()