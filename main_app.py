#Library imports
import numpy as np
import streamlit as st
import cv2
from model_architecture import build_model

st.set_page_config(
    page_title="Dog Breed Predictor",
    page_icon="🐶",
    layout="centered"
)

# Load model
model = build_model(num_classes=50)
model.load_weights("dog_breed_50.h5")

# Class names
CLASS_NAMES = [
    "labrador_retriever",
    "golden_retriever",
    "german_shepherd",
    "french_bulldog",
    "pug", "beagle", "siberian_husky", "pomeranian",
    "chihuahua", "yorkshire_terrier", "shih-tzu",
    "doberman", "boxer", "great_dane", "rottweiler",
    "border_collie", "basset", "maltese_dog",
    "irish_setter", "bernese_mountain_dog",
    "cocker_spaniel", "english_springer", "collie",
    "dachshund", "american_staffordshire_terrier",
    "staffordshire_bullterrier", "greyhound",
    "italian_greyhound", "miniature_schnauzer",
    "giant_schnauzer", "standard_schnauzer",
    "miniature_poodle", "standard_poodle", "toy_poodle",
    "dalmatian", "bluetick", "akita", "malamute",
    "samoyed", "keeshond", "whippet", "papillon",
    "pekinese", "pembroke", "norwegian_elkhound",
    "west_highland_white_terrier", "soft-coated_wheaten_terrier",
    "scottish_deerhound", "afghan_hound", "saluki"
]

# Title
st.title("Dog Breed Identification")

# Upload Image
uploaded = st.file_uploader("Upload an image of the dog...", type=["png", "jpg", "jpeg"])

if uploaded:
    # read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # preview
    st.image(img_cv, width=250)

    # identify button
    if st.button("Identify"):
        with st.spinner("Identifying..."):
            img = cv2.resize(img_cv, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            breed = CLASS_NAMES[np.argmax(pred)]

        st.success(f"🐶 The Dog Breed is **{breed.replace('_', ' ')}**")
