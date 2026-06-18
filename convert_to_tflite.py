import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH  = r"c:\Dara Cantik\SEC\Model_CNN_Agriwarn_Final.keras"
TFLITE_PATH = os.path.join(BASE_DIR, "data", "model_agriwarn.tflite")

print("Loading .keras model...")
model = tf.keras.models.load_model(KERAS_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"Selesai! Disimpan: {TFLITE_PATH}")
print(f"Ukuran: {size_mb:.1f} MB")
