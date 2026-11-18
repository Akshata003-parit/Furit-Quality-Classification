
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fruit Quality Classification",
    page_icon="🍎",
    layout="centered"
)

# Project heading
st.title("🍎 Fruit Quality Classification")
st.write("Deep Learning Based Fruit Quality Detection")
st.write("Upload an image to identify its fruit type and quality.")

# Fruit categories
class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_quality_model.keras")

model = load_model()

# Image upload
uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

# Predict button
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Fruit Image",
        width=300
    )

    if st.button("Predict Fruit Quality"):

        # Resize image
        image = image.resize((128, 128))

        # Convert image to array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # CNN prediction
        predictions = model.predict(image_array)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

        # Display results
        st.subheader("Prediction Result")

        if "fresh" in predicted_class:
            st.success("Fruit Quality: Fresh")
        else:
            st.error("Fruit Quality: Rotten")

        fruit_name = predicted_class.replace(
            "fresh", ""
        ).replace("rotten", "")

        st.write("Fruit Type:", fruit_name.capitalize())

        st.write(
            "Predicted Category:",
            predicted_class
        )

        st.write(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        st.progress(float(confidence / 100))
