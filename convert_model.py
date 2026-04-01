#!/usr/bin/env python
"""
Convert Keras 3.x .keras model format to Keras 2.x-compatible .h5 format
This resolves the deserialization error on Render (TensorFlow 2.13.1 uses Keras 2.x)
"""
import tensorflow as tf
import os

def convert_model():
    input_path = 'models/best_model.keras'
    output_path = 'models/final_model.h5'
    
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found")
        return False
    
    print(f"Loading model from {input_path}...")
    try:
        model = tf.keras.models.load_model(input_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    print(f"Saving to {output_path} in H5 format...")
    try:
        model.save(output_path, save_format='h5')
        print(f"✓ Model saved successfully")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        return False
    
    # Verify the conversion
    print(f"Verifying conversion...")
    try:
        verify_model = tf.keras.models.load_model(output_path)
        print(f"✓ Verification successful - H5 model loads correctly")
        return True
    except Exception as e:
        print(f"ERROR during verification: {e}")
        return False

if __name__ == '__main__':
    success = convert_model()
    exit(0 if success else 1)
