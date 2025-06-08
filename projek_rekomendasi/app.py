# app.py (FastAPI version)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pydantic import BaseModel

# Load model dan alat bantu
search_model = tf.keras.models.load_model("models/search_model_tf.keras")
user_model = tf.keras.models.load_model("models/user_model_tf.keras")

search_vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))
search_scaler = pickle.load(open("models/scaler.pkl", "rb"))

user_query_vectorizer = pickle.load(open("models/user_tfidf.pkl", "rb"))
user_product_vectorizer = pickle.load(open("models/product_tfidf.pkl", "rb"))
user_scaler = pickle.load(open("models/scaler_user.pkl", "rb"))

# Load data
product_data = pd.read_csv("../data/products.csv")

# Inisialisasi FastAPI
app = FastAPI()

# ===== Model Request ===== #
class SearchRequest(BaseModel):
    query: str

class UserRequest(BaseModel):
    recent_queries: List[str]  # 🔧 ini yang benar

# Endpoint untuk rekomendasi pencarian (berdasarkan query tunggal)
@app.post("/recommend")
def recommend(request: SearchRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Parameter 'query' diperlukan")

    tfidf = search_vectorizer.transform([query]).toarray()
    ratings = product_data['rating']
    sales = product_data['sales']
    scaled_features = search_scaler.transform(np.column_stack((ratings, sales)))

    X = np.hstack((np.repeat(tfidf, len(product_data), axis=0), scaled_features))
    predictions = search_model.predict(X).flatten()

    product_data['score'] = predictions
    top_products = product_data.sort_values('score', ascending=False).head(5)

    result = top_products[['title', 'score']].to_dict(orient='records')  # 🔧 ganti 'product_name' jadi 'title'
    return JSONResponse(content=result)

# Endpoint untuk rekomendasi berdasarkan 3 query terakhir user
@app.post("/recommend/user")
def recommend_user(request: UserRequest):
    if not request.recent_queries or len(request.recent_queries) < 1:
        raise HTTPException(status_code=400, detail="Parameter 'recent_queries' diperlukan minimal 1")

    combined_query = " ".join(request.recent_queries[-3:])  # 🔧 Ambil 3 terakhir, gabung jadi string

    user_query_tfidf = user_query_vectorizer.transform([combined_query]).toarray()
    product_tfidf = user_product_vectorizer.transform(product_data['title']).toarray()
    scaled_features = user_scaler.transform(product_data[['rating', 'sales']])

    X = np.hstack((np.repeat(user_query_tfidf, len(product_data), axis=0), product_tfidf, scaled_features))
    predictions = user_model.predict(X).flatten()

    product_data['score'] = predictions
    top_products = product_data.sort_values('score', ascending=False).head(5)

    result = top_products[['title', 'score']].to_dict(orient='records')
    return JSONResponse(content=result)
