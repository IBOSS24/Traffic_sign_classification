from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from Src.gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap
from Src.model_architecture import load_traffic_sign_model
from Src.preprocessing import preprocess_image
from Src.utils import load_class_names, top_k_predictions


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "traffic.weights.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


@st.cache_resource
def get_model():
    return load_traffic_sign_model(MODEL_PATH)


@st.cache_data
def get_class_names():
    return load_class_names(CLASS_NAMES_PATH)


st.set_page_config(page_title="Traffic Sign Classifier", layout="centered")

st.title("🚦 Traffic Road Sign Classifier")
st.write("Upload a traffic sign image and the model will predict it.")

with st.expander("About the model"):
    st.write(
        """
        This application uses a convolutional neural network trained on 43
        traffic sign classes. It provides real-time inference, confidence
        scoring, top-k predictions, and Grad-CAM explainability.
        """
    )

try:
    model = get_model()
    class_names = get_class_names()
except Exception as exc:
    st.error(f"Application setup failed: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    processed_img = preprocess_image(image)

    with st.spinner("Analyzing traffic sign..."):
        prediction = model.predict(processed_img, verbose=0)

    prediction_scores = prediction[0]
    class_id = int(np.argmax(prediction_scores))
    confidence = float(prediction_scores[class_id])

    st.subheader("Prediction")
    st.success(class_names[class_id])

    st.subheader("Confidence")
    st.progress(confidence)
    st.info(f"{confidence * 100:.2f}%")

    st.subheader("Top 3 predictions")
    for label, probability in top_k_predictions(prediction_scores, class_names, k=3):
        st.write(f"{label}: {probability * 100:.2f}%")

    st.subheader("Why the model predicted this")
    try:
        last_conv_layer = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer)
        result_img = overlay_heatmap(image, heatmap)
        st.image(result_img, caption="Model attention map", use_container_width=True)
        st.write("The highlighted regions contributed most strongly to the prediction.")
    except Exception as exc:
        st.warning(f"Grad-CAM could not be generated: {exc}")

    st.subheader("Top 5 predictions")
    top5 = top_k_predictions(prediction_scores, class_names, k=5)
    st.bar_chart({label: float(probability) for label, probability in top5})

    if confidence < 0.6:
        st.warning("Low confidence prediction. The model is unsure.")
    else:
        st.success("High confidence prediction.")
