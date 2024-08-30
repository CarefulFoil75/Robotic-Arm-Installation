import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# Load the pre-trained MobileNetV2 model with weights trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')
 
# Function to preprocess input images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 pixels
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Preprocess the image for MobileNetV2
    return img
 
# Function to decode the prediction
def decode_prediction(pred):
    return tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=1)[0]
 
# Function to classify an image
def classify_image(image_path):
    img = preprocess_image(image_path)  # Preprocess the image
    pred = model.predict(img)  # Make the prediction
    decoded_pred = decode_prediction(pred)  # Decode the prediction
    print(f"Predicted class: {decoded_pred[0][1]} with confidence {decoded_pred[0][2]:.2f}")
    return decoded_pred
 
# Main function to test the classification
if __name__ == '__main__':
    image_path = 'test_image.jpg'  # Replace with your image path
    prediction = classify_image(image_path)
 
    # Show the input image and the prediction
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {prediction[0][1]} ({prediction[0][2]*100:.2f}%)")
    plt.show()
 