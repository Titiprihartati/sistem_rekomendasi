# Product Recommendation System

Sistem rekomendasi produk berbasis machine learning menggunakan model TensorFlow dengan fitur TF-IDF dan normalisasi, serta API RESTful yang dibangun menggunakan FastAPI.

---

## Deskripsi

Proyek ini bertujuan untuk membangun sistem rekomendasi produk menggunakan dua model yang berbeda:

- **Model rekomendasi pencarian (`search_model_recomm.py`)**: Rekomendasi berdasarkan query pencarian tunggal.
- **Model rekomendasi pengguna (`user_model_recomm.py`)**: Rekomendasi berdasarkan gabungan beberapa query pencarian terakhir dari pengguna.

Model memanfaatkan fitur teks (TF-IDF) dari nama produk dan query pengguna, serta fitur numerik (rating dan penjualan) yang dinormalisasi.

API disediakan menggunakan FastAPI (`app.py`) untuk melayani permintaan rekomendasi produk.

---

## Struktur Proyek

├── data/
│ └── products.csv # Dataset produk (judul, rating, penjualan, dsb)
├── models/
│ ├── search_model_tf.keras
│ ├── user_model_tf.keras
│ ├── tfidf.pkl
│ ├── user_tfidf.pkl
│ ├── product_tfidf.pkl
│ ├── scaler.pkl
│ └── scaler_user.pkl
├── user_model_recomm.py # Script training model rekomendasi user
├── search_model_recomm.py # Script training model rekomendasi pencarian
├── app.py # API FastAPI untuk melayani request rekomendasi
└── README.md # Dokumentasi proyek

---

## Dataset

Dataset `products.csv` berisi minimal kolom berikut:

- `title` : Nama produk
- `rating` : Rating produk (numerik)
- `sales` : Jumlah penjualan produk (numerik)
- `query` : Query pencarian pengguna (hanya untuk user model)

---

## Cara Kerja

### 1. Pelatihan Model

- **search_model_recomm.py**
  - Membaca data produk
  - Menggunakan TF-IDF untuk mengubah nama produk jadi fitur numerik
  - Menggabungkan dengan fitur rating dan sales yang dinormalisasi
  - Melatih model Neural Network sederhana untuk klasifikasi dummy label (semua label 1)
  - Menyimpan model dan alat bantu (vectorizer, scaler)

- **user_model_recomm.py**
  - Membaca data produk dan query pencarian pengguna
  - Menggunakan TF-IDF pada query dan produk
  - Menggabungkan fitur tersebut dengan rating dan sales yang dinormalisasi
  - Melatih model Neural Network sederhana
  - Menyimpan model dan alat bantu

### 2. API FastAPI (`app.py`)

- Memuat model dan alat bantu (vectorizer & scaler)
- Menyediakan endpoint:

  - `/recommend`  
    Menerima query pencarian tunggal, mengembalikan 5 produk terbaik berdasarkan skor prediksi.

  - `/recommend/user`  
    Menerima list 1 sampai 3 query terakhir pengguna, mengembalikan 5 produk terbaik berdasarkan skor prediksi gabungan.

---

## Cara Menjalankan

### Prasyarat

- Python 3.8+
- Install dependencies:

```bash
pip install tensorflow scikit-learn pandas fastapi uvicorn

### Training Model 
python user_model_recomm.py
python search_model_recomm.py

### Jalankan API
uvicorn app:app --reload
