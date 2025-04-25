import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load EfficientNetB0 (using default 224x224 input)
model = EfficientNetB0(weights='imagenet')

def predict_dog_breed(img):
    """Predicts the dog breed from an image (using EfficientNet)."""
    img = img.resize((224, 224))  # Resizes the input image to 224x224 pixels, the expected size for EfficientNetB0.
    img_array = image.img_to_array(img)  # Converts the PIL Image object to a NumPy array.
    img_array = np.expand_dims(img_array, axis=0)  # Adds a batch dimension to the NumPy array, as the model expects a batch of images.
    img_array = preprocess_input(img_array)  # Preprocesses the image array according to EfficientNetB0's requirements.

    predictions = model.predict(img_array)  # Passes the preprocessed image to the EfficientNetB0 model for prediction.
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decodes the model's predictions into human-readable labels.

    result = []
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        result.append(f"{label}: {score * 100:.2f}%")  # Simple text with breed name and percentage
    return result

def main():
    st.title("Dog Breed Identification")
    st.write("Upload an image of a dog, and this app will predict its breed using a pre-trained model.")
    st.write("Click the **'Upload Image'** button below to upload a dog picture, and then click **'Predict Breed'** to see the prediction.")

    uploaded_file = st.file_uploader("Upload an image of a dog", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
       try:
            # Load image
            img = Image.open(uploaded_file)

            # Create two columns to display the image and the predictions side by side
            col1, col2 = st.columns([2, 1])  # Adjust column width ratio here

            with col1:
                # Resize the image to a smaller size (say, 400px wide)
                img = img.resize((400, int(400 * img.height / img.width)))
                st.image(img, caption="Uploaded Image", use_container_width=True)

            with col2:
                # Show image dimensions
                st.write(f"Uploaded Image Size: {img.size[0]} x {img.size[1]} pixels")

                # Prediction button with improved feedback
                if st.button("Predict Breed", key="predict-button"):
                    with st.spinner("Predicting..."):
                        predictions = predict_dog_breed(img)

                    # Display predictions with styled results
                    st.markdown('<p class="header">Prediction Results:</p>', unsafe_allow_html=True)
                    for prediction in predictions:
                        st.markdown(f"<p class='prediction-result'>{prediction}</p>", unsafe_allow_html=True)

       except Exception as e:
            st.error(f"An error occurred: {e}")
if __name__ == "__main__":
    main()