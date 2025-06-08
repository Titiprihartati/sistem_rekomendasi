# user_model_recomm.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 1. Baca data
data = pd.read_csv("../data/products.csv")

# 2. Kolom yang digunakan
texts = data['query']
products = data['title']
ratings = data['rating']
sales = data['sales']

# 3. TF-IDF vectorizer
query_vectorizer = TfidfVectorizer()
product_vectorizer = TfidfVectorizer()

query_tfidf = query_vectorizer.fit_transform(texts).toarray()
product_tfidf = product_vectorizer.fit_transform(products).toarray()

# 4. Normalisasi
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(np.column_stack((ratings, sales)))

# 5. Gabung fitur
X = np.hstack((query_tfidf, product_tfidf, scaled_features))

# 6. Dummy label
y = np.ones((len(X), 1))

# 7. Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Training
model.fit(X, y, epochs=5, batch_size=32)

# 9. Simpan model dan vectorizer
os.makedirs("models", exist_ok=True)
model.save("models/user_model_tf.keras")
pickle.dump(query_vectorizer, open("models/user_tfidf.pkl", "wb"))
pickle.dump(product_vectorizer, open("models/product_tfidf.pkl", "wb"))
pickle.dump(scaler, open("models/scaler_user.pkl", "wb"))
