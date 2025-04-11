import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_key.json")  # Ensure this file is in your project folder
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Save Chat Interaction
def save_chat(db, user, question, response):
    db.collection("chat_history").add({
        "user": user,
        "question": question,
        "response": response,
        "timestamp": datetime.now()
    })

# Save Crop Prediction
def save_crop_prediction(db, city, N, P, K, temp, humidity, ph, rainfall, prediction):
    db.collection("crop_predictions").add({
        "city": city,
        "N": N,
        "P": P,
        "K": K,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "prediction": prediction,
        "timestamp": datetime.now()
    })

# Save Leaf Disease Result
def save_leaf_disease(db, filename, predicted_class):
    db.collection("leaf_detections").add({
        "filename": filename,
        "prediction": predicted_class,
        "timestamp": datetime.now()
    })

# (Optional) Fetch Chat History
def get_chat_history(db, limit=10):
    docs = db.collection("chat_history")\
             .order_by("timestamp", direction=firestore.Query.DESCENDING)\
             .limit(limit).stream()
    return [doc.to_dict() for doc in docs]
