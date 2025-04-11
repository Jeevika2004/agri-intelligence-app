import requests
import joblib

# Load model and encoder
model = joblib.load('crop_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# --- Fetch live weather data ---
API_KEY = '3072c0ce66324f7d4fb269e6faa7fec6'  # Make sure this is correct
CITY = 'Coimbatore'

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
response = requests.get(url)
data = response.json()

# Debug response
print("📦 API Response:", data)

if 'main' not in data:
    print(f"❌ Failed to fetch weather data. Error: {data.get('message', 'Unknown error')}")
    exit()

# Extract required weather features
temperature = data['main']['temp']
humidity = data['main']['humidity']

# Hardcoded example values for other features
sample_input = {
    'N': 90,
    'P': 40,
    'K': 40,
    'temperature': temperature,
    'humidity': humidity,
    'ph': 6.5,
    'rainfall': 200.0
}

# Prepare input for prediction
features = [[
    sample_input['N'],
    sample_input['P'],
    sample_input['K'],
    sample_input['temperature'],
    sample_input['humidity'],
    sample_input['ph'],
    sample_input['rainfall']
]]

# Predict
predicted_label_encoded = model.predict(features)[0]
predicted_crop = label_encoder.inverse_transform([predicted_label_encoded])[0]

print(f"🌾 Recommended Crop for {CITY} (Live Weather): {predicted_crop}")

