import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io

# Load EfficientNetB0 (using default 224x224 input)
model = EfficientNetB0(weights='imagenet')
def predict_dog_breed(img):
    """Predicts the dog breed from an image (using EfficientNet)."""
    img = img.resize((224, 224)) # Resizes the input image to 224x224 pixels, the expected size for EfficientNetB0.
    img_array = image.img_to_array(img) # Converts the PIL Image object to a NumPy array.
    img_array = np.expand_dims(img_array, axis=0) # Adds a batch dimension to the NumPy array, as the model expects a batch of images.
    img_array = preprocess_input(img_array) # Preprocesses the image array according to EfficientNetB0's requirements.

    predictions = model.predict(img_array) # Passes the preprocessed image to the EfficientNetB0 model for prediction.
    decoded_predictions = decode_predictions(predictions, top=1)[0] # Decodes the model's predictions into human-readable labels, retrieving only the top prediction.

    if decoded_predictions: # Checks if the prediction list is not empty.
        return decoded_predictions[0][1] # Returns the breed name from the top prediction.
    else:
        return "Could not predict" # Returns a message if no prediction could be made.

def main():
    st.title("Dog Breed Identification") # Sets the title of the Streamlit app.

    uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"]) # Creates a file uploader widget that accepts JPG, JPEG, and PNG images.

    if uploaded_file is not None: # Checks if a file has been uploaded.
        try:
            img = Image.open(uploaded_file) # Opens the uploaded image using PIL.
            st.image(img, caption="Uploaded Image", use_container_width=True) # Displays the uploaded image in the Streamlit app.

            st.write(f"Uploaded Image Size: {img.size[0]} x {img.size[1]} pixels") # Displays the original size of the uploaded image.

            if st.button("Predict Breed"): # Creates a button that triggers breed prediction.
                with st.spinner("Predicting..."): # Displays a spinner while the prediction is being made.
                    prediction = predict_dog_breed(img) # Calls the predict_dog_breed function to get the prediction.
                st.subheader("Prediction Result:") # Displays the prediction result.
                st.write(prediction) # Writes the prediction to the streamlit app.
        except Exception as e: # Handles potential errors during image processing or prediction.
            st.error(f"An error occurred: {e}") # Displays an error message.

if __name__ == "__main__":
    main() # Runs the main function when the script is executed.