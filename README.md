Skin Disease Detection
----------------------

A Deep Learning-powered web application to detect skin diseases from images.

Overview
--------

This project uses a convolutional neural network (CNN) to classify skin diseases from images.
The model is trained on a dataset of images of various skin diseases and can predict the disease with a high degree of accuracy.

Features
--------

Detects 9 different skin diseases:
Actinic Keratosis
Atopic Dermatitis
Squamous Cell Carcinoma
Benign Keratosis
Dermatofibroma
Melanoma
Melanocytic Nevus
Vascular Lesion
Tinea Ringworm Candidiasis
Provides a list of required medicines for each disease

Requirements
------------

Python
TensorFlow
scikit-learn
Streamlit
pandas
NumPy

Installation
------------

Clone the repository: git clone https://github.com/your-username/skin-disease-detection.git
Install the requirements: pip install -r requirements.txt
Run the application: streamlit run app.py

Usage
-----

Upload an image of the skin disease
Click the "Detect" button to predict the disease
View the predicted disease and required medicines also.

Dataset Source
---------------

Dataset provider:[please click the link and download the data set here](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset/data)

files explaination
------------------
[1].requirements.txt:
A requirements.txt file is a text file that lists the dependencies required to run a this project. 
It specifies the packages that need to be installed in order to execute the project.
[2].app.py:
An app.py file is a Python script that serves as the main application file for an this project. 
It's responsible for loading the trained model, making predictions, and providing an interface for users to interact with the model.
[3].skin_detect.ipynb:
A .ipynb file is a notebook file used by Jupyter Notebook, a web-based interactive computing environment.
