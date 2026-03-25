import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def save_sample_digits():
    print("Loading MNIST to generate sample images...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    sample_dir = 'sample_images'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        
    # Save one of each digit (0-9)
    saved_digits = set()
    for i in range(len(x_test)):
        digit = y_test[i]
        if digit not in saved_digits:
            img_path = os.path.join(sample_dir, f'digit_{digit}.png')
            plt.imsave(img_path, x_test[i], cmap='gray')
            saved_digits.add(digit)
            if len(saved_digits) == 10:
                break
    
    print(f"Successfully saved 10 sample images (0-9) to the '{sample_dir}' folder.")

if __name__ == "__main__":
    save_sample_digits()
