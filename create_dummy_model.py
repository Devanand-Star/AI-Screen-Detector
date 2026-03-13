import tensorflow as tf
import numpy as np
import os

# Create a simple model that mimics the one in train.py
def create_dummy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Save it to the expected path
    model_path = os.path.join(os.path.dirname(__file__), "model", "screen_guard_model.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ Dummy model created at {model_path}")

if __name__ == "__main__":
    create_dummy_model()
