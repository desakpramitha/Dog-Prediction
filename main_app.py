# === Streamlit Dog Breed Predictor App (with simple explanations) ===
# Import library yang dibutuhkan
import numpy as np
import streamlit as st
import cv2
from model_architecture import build_model

# Set tampilan halaman Streamlit
st.set_page_config(
    page_title="Dog Breed Predictor",  # Judul tab
    page_icon="🐶",                   # Icon tab
    layout="centered"                 # Layout tengah
)

# Load model machine learning
model = build_model(num_classes=50)         # Membuat arsitektur model
model.load_weights("dog_breed_50.h5")      # Memuat bobot model hasil training

# Daftar nama ras anjing (urutan sesuai output model)
CLASS_NAMES = [
    "labrador_retriever", "golden_retriever", "german_shepherd", "french_bulldog",
    "pug", "beagle", "siberian_husky", "pomeranian", "chihuahua",
    "yorkshire_terrier", "shih-tzu", "doberman", "boxer", "great_dane",
    "rottweiler", "border_collie", "basset", "maltese_dog", "irish_setter",
    "bernese_mountain_dog", "cocker_spaniel", "english_springer", "collie",
    "dachshund", "american_staffordshire_terrier", "staffordshire_bullterrier",
    "greyhound", "italian_greyhound", "miniature_schnauzer", "giant_schnauzer",
    "standard_schnauzer", "miniature_poodle", "standard_poodle", "toy_poodle",
    "dalmatian", "bluetick", "akita", "malamute", "samoyed", "keeshond",
    "whippet", "papillon", "pekinese", "pembroke", "norwegian_elkhound",
    "west_highland_white_terrier", "soft-coated_wheaten_terrier",
    "scottish_deerhound", "afghan_hound", "saluki"
]

# Judul aplikasi
st.title("Dog Breed Identification")

# Session state untuk menyimpan status aplikasi
if "loading" not in st.session_state:
    st.session_state.loading = False  # Menandakan apakah sedang memproses

if "result" not in st.session_state:
    st.session_state.result = None    # Menyimpan hasil prediksi

if "opencv_image" not in st.session_state:
    st.session_state.opencv_image = None  # Menyimpan gambar yang sudah diproses

# Komponen upload file gambar
dog_image = st.file_uploader(
    "Upload an image of the dog...",       # Teks di UI
    type=["png", "jpg", "jpeg"],         # Format valid
    disabled=st.session_state.loading       # Disable jika sedang memproses
)

# Reset jika tidak ada gambar
if dog_image is None:
    st.session_state.result = None
    st.session_state.opencv_image = None

# Jika gambar di-upload
if dog_image is not None:
    file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)  # Membaca file
    opencv_image = cv2.imdecode(file_bytes, 1)                           # Decode ke OpenCV
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)         # Konversi warna
    st.session_state.opencv_image = opencv_image                         # Simpan gambar

    st.image(opencv_image, channels="RGB", width=250)  # Tampilkan gambar di layar

    # Tombol identifikasi
    identify_clicked = st.button(
        "Identify",                          # Label tombol
        disabled=st.session_state.loading      # Disable saat memproses
    )

    # Jika tombol diklik, aktifkan proses
    if identify_clicked:
        st.session_state.loading = True

# Jika sedang memproses, lakukan prediksi
if st.session_state.loading:
    with st.spinner("Identifying..."):
        img = cv2.resize(st.session_state.opencv_image, (224, 224))  # Resize ke ukuran input model
        img = img / 255.0                                            # Normalisasi
        img = np.expand_dims(img, axis=0)                            # Tambah batch dimensi

        preds = model.predict(img)                                   # Jalankan prediksi
        st.session_state.result = CLASS_NAMES[np.argmax(preds)]       # Pilih ras dengan skor tertinggi

    st.session_state.loading = False

# Tampilkan hasil prediksi
if st.session_state.result:
    st.title(
        f"The Dog Breed is {st.session_state.result.replace('_', ' ').title()}"  # Format teks hasil
    )