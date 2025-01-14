import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and labels
model = load_model('skindisease_model.h5')
labels = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion']

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img

# Function to make predictions
def make_prediction(image):
    img = np.expand_dims(image, axis=0)
    prediction = model.predict(img)
    return labels[np.argmax(prediction)]

# Main function
def main():
    st.title("Skin Disease Detection Application")
    st.write("This is a Deep Learning application to detect skin diseases from images.")

    # Upload image
    image_file = st.file_uploader("Upload an image of the skin lesion",
type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        # Load and display the image
        image = load_image(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        if st.button("Detect"):
            prediction = make_prediction(image)
            st.success(f"The predicted skin disease is: {prediction}")

if __name__ == "__main__":
    main()
     