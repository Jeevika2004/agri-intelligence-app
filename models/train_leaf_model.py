import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 1. Load your training/validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'data/archive/plantvillage dataset/color',       # ✅ updated path
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # ✅ for multi-class classification
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/archive/plantvillage dataset/color',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. Build a CNN model for multi-class classification
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')  # ✅ output units = num classes
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',   # ✅ updated loss
              metrics=['accuracy'])

# 4. Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# 5. Save the model
model.save("leaf_disease_model.h5")
print("✅ Model saved as leaf_disease_model.h5")
