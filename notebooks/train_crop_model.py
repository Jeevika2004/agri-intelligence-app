import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving model and encoder

# Step 1: Load Data
df = pd.read_csv('D:/agri_intelligence/data/Crop_recommendation.csv')

# Step 2: Encode Labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Step 3: Feature-Target Split
X = df.drop(['label', 'label_encoded'], axis=1)
y = df['label_encoded']

# Step 4: Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
print("✅ Model trained successfully!")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save Model and Label Encoder
joblib.dump(model, 'crop_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\n💾 Model and encoder saved as 'crop_model.pkl' and 'label_encoder.pkl'")
