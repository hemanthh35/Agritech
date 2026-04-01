"""
GOOGLE COLAB CONVERSION SCRIPT
Run this in a new Google Colab cell to convert best_model.keras to .h5 format

Instructions:
1. Open Google Colab: https://colab.research.google.com
2. Create a new cell and paste this code
3. Run it
4. Download the final_model.h5 file
5. Add to your GitHub repo and push
"""

# Step 1: Upload the .keras file to Colab
from google.colab import files
print("Upload best_model.keras file:")
uploaded = files.upload()

if 'best_model.keras' in uploaded:
    print("✓ File uploaded successfully")
else:
    print("✗ Please upload best_model.keras")
    exit()

# Step 2: Install/verify TensorFlow with Keras 3.x support
import tensorflow as tf
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Keras backend: {tf.keras.backend.backend()}")

# Step 3: Load the model (Keras 3.x can read it)
print("\nLoading best_model.keras...")
try:
    model = tf.keras.models.load_model('best_model.keras')
    print(f"✓ Model loaded successfully")
    print(f"  Type: {type(model).__name__}")
    print(f"  Input shape: {model.input_shape}")
except Exception as e:
    print(f"✗ Error loading: {e}")
    exit()

# Step 4: Save in .h5 format (compatible with Keras 2.x)
print("\nSaving as final_model.h5...")
try:
    model.save('final_model.h5', save_format='h5')
    print(f"✓ Model saved successfully")
except Exception as e:
    print(f"✗ Error saving: {e}")
    exit()

# Step 5: Verify the .h5 is readable
print("\nVerifying .h5 file...")
try:
    verify = tf.keras.models.load_model('final_model.h5')
    print(f"✓ H5 model loads correctly on Keras 3.x")
    print(f"  File ready for Keras 2.x deployment")
except Exception as e:
    print(f"✗ Verification failed: {e}")
    exit()

# Step 6: Download the converted file
import os
if os.path.exists('final_model.h5'):
    print("\n✓ final_model.h5 ready for download")
    files.download('final_model.h5')
    print("  Check your Downloads folder")
