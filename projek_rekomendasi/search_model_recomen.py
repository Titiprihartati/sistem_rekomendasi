# search_model_recomm.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 1. Baca data produk
data = pd.read_csv("../data/products.csv")

# 2. Ambil kolom yang dibutuhkan
names = data['title']
ratings = data['rating']
sales = data['sales']

# 3. TF-IDF pada nama produk
vectorizer = TfidfVectorizer()
product_tfidf = vectorizer.fit_transform(names).toarray()

# 4. Normalisasi rating dan penjualan
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(np.column_stack((ratings, sales)))

# 5. Gabungkan TF-IDF dan fitur numerik
X = np.hstack((product_tfidf, scaled_features))

# 6. Label dummy: kita anggap semua produk layak direkomendasikan (label = 1)
y = np.ones((len(X), 1))

# 7. Bangun model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Latih model
model.fit(X, y, epochs=5, batch_size=32)

# 9. Simpan model dan vektorizer
model.save("models/search_model_tf.keras")
pickle.dump(vectorizer, open("models/tfidf.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
