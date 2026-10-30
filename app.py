
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# =====================================================
# 2. PATHS AND SETTINGS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "waste_classifier_v4.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Plastic",
    "Paper",
    "Metal",
    "Organic"
]

# =====================================================
# 3. CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #16834a;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 16px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e9f7ef;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">♻️ Smart Waste Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered waste identification and disposal guidance</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# 4. LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# =====================================================
# 5. PREPROCESS IMAGE
# =====================================================
def preprocess_image(image):
    image = ImageOps.fit(
        image.convert("RGB"),
        IMG_SIZE,
        method=Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Model performs its own Rescaling layer.
    return np.expand_dims(image_array, axis=0)

# =====================================================
# 6. DISPOSAL RECOMMENDATIONS
# =====================================================
RECOMMENDATIONS = {
    "Plastic": {
        "icon": "🧴",
        "description": "Plastic waste",
        "advice": (
            "Separate plastic items from other waste. "
            "Clean and dry recyclable plastics before "
            "placing them in an appropriate recycling bin."
        )
    },

    "Paper": {
        "icon": "📄",
        "description": "Paper and cardboard waste",
        "advice": (
            "Keep paper and cardboard clean and dry. "
            "Separate them for paper recycling. "
            "Avoid mixing wet or food-contaminated paper."
        )
    },

    "Metal": {
        "icon": "🥫",
        "description": "Metal waste",
        "advice": (
            "Separate metal cans and other recyclable "
            "metal items. Use an appropriate metal "
            "recycling collection facility."
        )
    },

    "Organic": {
        "icon": "🍃",
        "description": "Biodegradable waste",
        "advice": (
            "Separate suitable food scraps and garden "
            "waste. Compost them where possible."
        )
    }
}

# =====================================================
# 7. IMAGE UPLOAD
# =====================================================
st.subheader("Upload Waste Image")

uploaded_file = st.file_uploader(
    "Choose an image of waste",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Waste Image",
            width="stretch"
        )

        if st.button(
            "🔍 Classify Waste",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("AI is analyzing the image..."):

                model = load_model()

                input_data = preprocess_image(image)

                predictions = model.predict(
                    input_data,
                    verbose=0
                )[0]

                if len(predictions) != len(CLASS_NAMES):
                    raise ValueError(
                        "Model output does not match the four waste classes."
                    )

                predicted_index = int(np.argmax(predictions))

                category = CLASS_NAMES[predicted_index]

                confidence = float(
                    predictions[predicted_index]
                ) * 100

            # =========================================
            # 8. DISPLAY RESULTS
            # =========================================
            result = RECOMMENDATIONS[category]

            st.divider()
            st.subheader("Classification Result")

            st.markdown(
                f"""
                <div class="result-box">
                    <h1>{result['icon']}</h1>
                    <h2>{category}</h2>
                    <p>{result['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                max(0.0, min(confidence / 100, 1.0))
            )

            st.subheader("♻️ Disposal Recommendation")

            st.info(result["advice"])

            st.caption(
                "Note: AI predictions are estimates. "
                "Always check local waste segregation rules."
            )

    except Exception as error:
        st.error("Unable to classify this image.")
        st.error(str(error))

# =====================================================
# 9. FOOTER
# =====================================================
st.divider()

st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Smart Waste Classification System | "
    "Powered by TensorFlow and MobileNetV2"
    "</div>",
    unsafe_allow_html=True
)
