import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load the model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Streamlit app
st.title("Image Caption Generator")
st.write("Upload an image, and the app will generate a caption describing the image.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption button
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            try:
                # Preprocess the image
                inputs = processor(images=image, return_tensors="pt")

                # Generate caption
                outputs = model.generate(**inputs)
                caption = processor.decode(outputs[0], skip_special_tokens=True)

                # Display the caption
                st.success("Caption Generated!")
                st.write("**Generated Caption:**")
                st.write(caption)

            except Exception as e:
                st.error(f"An error occurred: {e}")
