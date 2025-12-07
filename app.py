import streamlit
import tensorflow
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and labels
model = load_model('skin_disease.h5', compile=False)
labels = ['Actinic Keratosis', 'Atopic Dermatitis','Squamous Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion','Tinea Ringworm Candidiasis']

# Dictionary of skin diseases and their required medicines
medicines = {
    'Actinic Keratosis': ['Imiquimod cream', 'Fluorouracil cream', 'Ingenol mebutate gel'],
    'Atopic Dermatitis': ['Topical corticosteroids', 'Topical immunomodulators', 'Oral antihistamines'],
    'Squamous Cell Carcinoma': ['Surgical excision', 'Mohs surgery', 'Radiation therapy', 'Chemotherapy'],
    'Benign Keratosis': ['Cryotherapy', 'Cantharidin', 'Salicylic acid', 'Urea cream'],
    'Dermatofibroma': ['Surgical excision', 'Cryotherapy', 'Intralesional steroid injections'],
    'Melanoma': ['Surgical excision', 'Immunotherapy', 'Targeted therapy', 'Chemotherapy'],
    'Melanocytic Nevus': ['Surgical excision', 'Laser therapy', 'Cryotherapy'],
    'Vascular Lesion': ['Laser therapy', 'Sclerotherapy', 'Surgical excision'],
    'Tinea Ringworm Candidiasis': ['Topical antifungals', 'Oral antifungals', 'Antifungal creams']
}

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
    image_file = st.file_uploader("Upload an image of the skin disease", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        # Load and display the image
        image = load_image(image_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Make prediction
        if st.button("Detect"):
            prediction = make_prediction(image)
            st.success(f"The predicted skin disease is: {prediction}")
            st.success("Please use the following medicines and ointments for better results:")
            
            # Display required medicines one by one
            medicine_list = medicines[prediction]
            for i, medicine in enumerate(medicine_list):
                st.write(f"Medicine {i+1}: {medicine}")

if __name__ == "__main__":
    main()
