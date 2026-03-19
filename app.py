import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("best_model.keras", compile=False)

model = load_my_model()

# Class names
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# UI
st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            pred = model.predict(img_array)

        pred_index = np.argmax(pred)
        confidence = np.max(pred)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.subheader("Prediction Result")
        st.success(class_names[pred_index])
        st.write("Confidence:", f"{confidence*100:.2f}%")

    except Exception as e:
        st.error("Error processing image")