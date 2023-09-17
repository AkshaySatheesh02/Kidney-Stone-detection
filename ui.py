from fastai.vision.all import *
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import numpy as np
import fastbook

import streamlit as st
import cv2
import fastai.data.all as fa_data
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from fastai.learner import load_learner

import base64

path = Path(r"C:\Users\aksha\Desktop\New folder\DIP PROJ\Kidney_stone_detection-main\Dataset")
all_files = get_image_files(path)
augs = [RandomResizedCropGPU(size=224, min_scale=0.75), Rotate(), Zoom()]
dblock = DataBlock(blocks=(ImageBlock(cls=PILImage), CategoryBlock),
                   splitter=GrandparentSplitter(train_name='Train', valid_name='Test'),
                   get_y=parent_label,
                   item_tfms=Resize(512, method="squish"),
                   batch_tfms=augs,
                   )

dls_test = dblock.dataloaders(all_files)
# Load the trained model
model = create_cnn_model(xresnet50, n_out=2, pretrained=False)
learn1 = learn1 = Learner(dls_test, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy).load(path/"../models/kidney-50")


# Replace <path_to_model> with the path where your trained model is saved

def segment_kidney(image, threshold_level=1.5):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to a supported data type (e.g., uint8)
    image = image.astype(np.uint8)

    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = threshold_level * _
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)
    segmented_image = cv2.bitwise_and(image, inverted_mask)

    return segmented_image



def perform_segmentation(uploaded_image):
    # Read the uploaded image file
    image = Image.open(uploaded_image)
    image_array = np.array(image)

    # Perform kidney segmentation on the image array
    segmented_image = segment_kidney(image_array)
    segmented_images = [segmented_image]
    return segmented_images



def predict_image(image):
    # Preprocess the image and make a prediction using the loaded model
    img = Image.open(image).convert('RGB')
    prediction, _, probabilities = learn1.predict(img)
    return prediction, _, probabilities

def main():
    st.title("Kidney Stone Detection")
    st.sidebar.title("Options")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Upload image
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display image caption
        caption_text = "Uploaded Image"
        caption_style = "font-size: 55px; font-weight: bold; text-align: center; color: black;"
        st.caption(f"<p style='{caption_style}'>{caption_text}</p>", unsafe_allow_html=True)

        # Display uploaded image
        image_data = uploaded_image.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        st.markdown(
            f"<div style='display: flex; justify-content: center;'><img src='data:image/jpeg;base64,{encoded_image}' style='max-width: 60%;'></div>",
            unsafe_allow_html=True
        )

        # Perform kidney segmentation and display segmented images
        #segmented_images = perform_segmentation(uploaded_image)
        #display_segmented_images(segmented_images)

        # Make prediction on the uploaded image
        #prediction, _, probabilities = predict_image(uploaded_image)
        #st.subheader("Prediction")
        #st.write("Predicted class:", prediction)
        # Perform kidney segmentation and display segmented images
        segmented_images = perform_segmentation(uploaded_image)
        display_segmented_images(segmented_images)

        # Make predictions on segmented images
        predictions = []
        probabilities = []
        for segmented_image in segmented_images:
            prediction, _, probs = predict_segmented_image(segmented_image)
            predictions.append(prediction)
            probabilities.append(probs)

        # Display predictions
        st.subheader("Predictions")
        for i, prediction in enumerate(predictions):
            st.write(f"Segmented Image {i+1}: Predicted class - {prediction}")






def display_segmented_images(segmented_images):
    # Create subplots dynamically based on the number of segmented images
    num_images = len(segmented_images)
    fig, axs = plt.subplots(1, num_images, figsize=(3, 1.5))

    # Iterate over the segmented images and display them
    for i, image in enumerate(segmented_images):
        ax = axs[i] if num_images > 1 else axs  # Handle case with a single segmented image
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Segmented Image", fontsize=10)  # Add segmented image title and increase font size

    plt.tight_layout()
    st.pyplot(fig)

    #-------------------------
def predict_segmented_image(segmented_image):
    # Preprocess the segmented image and make a prediction using the loaded model
    img = Image.fromarray(segmented_image).convert('RGB')
    prediction, _, probabilities = learn1.predict(img)
    return prediction, _, probabilities





if __name__ == '__main__':
    main()