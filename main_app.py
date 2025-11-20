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

# Load the model only once
@st.cache_resource
def load_model():
    model = build_model(num_classes=50)
    model.load_weights("dog_breed_50.h5")
    return model

model = load_model()

# Class names
CLASS_NAMES = [
    "labrador_retriever", "golden_retriever", "german_shepherd", "french_bulldog",
    "pug", "beagle", "siberian_husky", "pomeranian", "chihuahua", "yorkshire_terrier",
    "shih-tzu", "doberman", "boxer", "great_dane", "rottweiler", "border_collie",
    "basset", "maltese_dog", "irish_setter", "bernese_mountain_dog",
    "cocker_spaniel", "english_springer", "collie", "dachshund",
    "american_staffordshire_terrier", "staffordshire_bullterrier", "greyhound",
    "italian_greyhound", "miniature_schnauzer", "giant_schnauzer",
    "standard_schnauzer", "miniature_poodle", "standard_poodle", "toy_poodle",
    "dalmatian", "bluetick", "akita", "malamute", "samoyed", "keeshond", "whippet",
    "papillon", "pekinese", "pembroke", "norwegian_elkhound",
    "west_highland_white_terrier", "soft-coated_wheaten_terrier",
    "scottish_deerhound", "afghan_hound", "saluki"
]

st.title("Dog Breed Identification")

# Initialize state
if "loading" not in st.session_state:
    st.session_state.loading = False

if "result" not in st.session_state:
    st.session_state.result = None

if "opencv_image" not in st.session_state:
    st.session_state.opencv_image = None

# Upload image
dog_image = st.file_uploader(
    "Upload an image of the dog...",
    type=["png", "jpg", "jpeg"],
    disabled=st.session_state.loading
)

# Reset result if file removed
if dog_image is None:
    st.session_state.result = None
    st.session_state.opencv_image = None

# If file uploaded
if dog_image is not None:

    # Read file
    file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    if opencv_image is None:
        st.error("Failed to read the image. Try another file.")
    else:
        # Convert and store image
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        st.session_state.opencv_image = opencv_image

        # Show image
        st.image(opencv_image, channels="RGB", width=250)

        # Identify button
        if st.button("Identify", disabled=st.session_state.loading):
            st.session_state.loading = True

# If loading → run prediction
if st.session_state.loading:

    if st.session_state.opencv_image is None:
        st.error("Image is missing. Please upload again.")
        st.session_state.loading = False
    else:
        with st.spinner("Identifying..."):
            img = cv2.resize(st.session_state.opencv_image, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            st.session_state.result = CLASS_NAMES[np.argmax(preds)]

        st.session_state.loading = False

# Show result
if st.session_state.result:
    st.success(f"The Dog Breed is **{st.session_state.result.replace('_', ' ')}**")