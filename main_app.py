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

#Loading the Model
model = build_model(num_classes=50)
model.load_weights("dog_breed_50.h5")

#Name of Classes
CLASS_NAMES = [
    "labrador_retriever",
    "golden_retriever",
    "german_shepherd",
    "french_bulldog",
    "pug",
    "beagle",
    "siberian_husky",
    "pomeranian",
    "chihuahua",
    "yorkshire_terrier",
    "shih-tzu",
    "doberman",
    "boxer",
    "great_dane",
    "rottweiler",
    "border_collie",
    "basset",
    "maltese_dog",
    "irish_setter",
    "bernese_mountain_dog",
    "cocker_spaniel",
    "english_springer",
    "collie",
    "dachshund",      # dataset tidak punya → diganti dengan yang paling dekat
    "american_staffordshire_terrier",
    "staffordshire_bullterrier",
    "greyhound",      # dataset tidak punya → diganti dengan italian_greyhound
    "italian_greyhound",
    "miniature_schnauzer",
    "giant_schnauzer",
    "standard_schnauzer",
    "miniature_poodle",
    "standard_poodle",
    "toy_poodle",
    "dalmatian",      # dataset tidak punya → gunakan bluetick (closest)
    "bluetick",
    "akita",          # dataset tidak punya → gunakan malamute (closest)
    "malamute",
    "samoyed",
    "keeshond",
    "whippet",
    "papillon",
    "pekinese",
    "pembroke",
    "norwegian_elkhound",
    "west_highland_white_terrier",
    "soft-coated_wheaten_terrier",
    "scottish_deerhound",
    "afghan_hound",
    "saluki"
]

# Setting Title of App
st.title("Dog Breed Identification")

# Inisialisasi state
if "loading" not in st.session_state:
    st.session_state.loading = False

if "result" not in st.session_state:
    st.session_state.result = None



#Uploading the dog image
dog_image = st.file_uploader("Upload an image of the dog...", type=["png", "jpg", "jpeg"], disabled=st.session_state.loading)

# Kalau file dihapus/reset → hapus hasil prediction
if dog_image is None:
    st.session_state.result = None

# Jika ada file → tampilkan gambar + tombol identify
if dog_image is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Convert BGR to RGB
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Display image
    st.image(opencv_image, channels="RGB", width=250)

    #  Tombol Identify → akan disable kalau sedang loading
    identify_clicked = st.button(
        "Identify",
        disabled=st.session_state.loading
    )

    # Kalau tombol diklik → mulai loading
    if identify_clicked:
        st.session_state.loading = True
        st.rerun()


# Saat loading → jalankan prediction
if st.session_state.loading:

    with st.spinner("Identifying..."):
        img = cv2.resize(opencv_image, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        Y_pred = model.predict(img)
        st.session_state.result = CLASS_NAMES[np.argmax(Y_pred)]

    # Matikan loading dan tampilkan hasil
    st.session_state.loading = False
    st.rerun()


# Kalau sudah ada hasil → tampilkan
if st.session_state.result:
    st.title(f"The Dog Breed is {st.session_state.result.replace('_', ' ')}")