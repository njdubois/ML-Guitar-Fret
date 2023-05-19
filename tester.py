import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('guitar_fret_model.h5')

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB to remove alpha channel
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

image_path = 'fret_1_strings_6_8.png'  # Replace with the path to your input image
input_image = preprocess_image(image_path)

# Use the model to predict the frets in the image
predictions = model.predict(input_image)

# Visualize the predicted frets
def visualize_frets(image_path, predictions):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.grid(False)

    # Add marks on the image for predicted frets
    width, height = image.size
    fret_width = width // 8
    fret_height = height // 8
    for i, prediction in enumerate(predictions[0]):
        if prediction >= 0.1:  # Adjust the threshold as needed
            fret_num = i % 8
            string_num = i // 8
            x = fret_num * fret_width
            y = string_num * fret_height
            rect = plt.Rectangle((x, y), fret_width, fret_height,
                                 edgecolor='red', facecolor='none', linewidth=2)
            plt.gca().add_patch(rect)

    plt.show()

visualize_frets(image_path, predictions)

