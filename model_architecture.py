# === Model Architecture (Dengan Penjelasan Sederhana & Jelas) ===
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# Fungsi untuk membangun model klasifikasi ras anjing
# - num_classes: jumlah kategori ras yang ingin diprediksi

def build_model(num_classes=50):
    # Mengambil MobileNetV2 sebagai "backbone" (model dasar)
    # MobileNetV2 dipilih karena ringan, cepat, dan sudah dilatih pada ImageNet
    base_model = MobileNetV2(
        weights='imagenet',  # Menggunakan bobot pretrained ImageNet
        include_top=False,  # Bagian classifier bawaan dihapus
        input_shape=(224, 224, 3)  # Ukuran input gambar
    )

    # Membekukan sebagian besar layer MobileNetV2
    # Tujuan:
    # - Agar proses training lebih cepat
    # - Menghindari overfitting karena dataset anjing relatif kecil
    # Model hanya melatih 20 layer paling atas (bagian paling akhir)
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Mengambil output fitur dari MobileNetV2
    x = base_model.output

    # Mengubah feature map 7x7 menjadi satu vector
    # (Meringkas informasi visual penting)
    x = GlobalAveragePooling2D()(x)

    # Dropout untuk mengurangi overfitting
    x = Dropout(0.3)(x)

    # Layer dense yang belajar pola detail untuk membedakan ras
    x = Dense(256, activation='relu')(x)

    # Dropout tambahan agar model lebih stabil
    x = Dropout(0.2)(x)

    # Layer output — menghasilkan probabilitas untuk setiap ras
    preds = Dense(num_classes, activation='softmax')(x)

    # Gabungkan input MobileNetV2 + custom classifier
    model = Model(inputs=base_model.input, outputs=preds)

    return model