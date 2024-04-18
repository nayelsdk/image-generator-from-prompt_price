import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import io
import configparser

# Load the configuration from a config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# Attempt to retrieve the Hugging Face API token from the configuration file
hf_token = config.get('DEFAULT', 'HUGGINGFACE_TOKEN', fallback=None)
if not hf_token:
    st.error("Please configure your Hugging Face API token in the 'config.ini' file under the [DEFAULT] section with the HUGGINGFACE_TOKEN key.")
    st.stop()

# Setup the title and input box on the Streamlit UI
st.title("Image Generator from Text - Using Hugging Face Models")
text_input = st.text_input("Enter the text prompt you wish to transform into an image:", "")

# Only proceed if the user has entered some text
if text_input:
    try:
        # Check if CUDA is available and utilize it; otherwise, use the CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Correctly load the pipeline by passing keyword arguments
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id="CompVis/stable-diffusion-v1-4",  # Changed to keyword argument for clarity
            revision="fp16",                          # Example revision (use appropriate one)
            torch_dtype=torch.float16
        ).to(device)
        
        # Set a manual seed for reproducibility
        generator = torch.Generator(device).manual_seed(42)

        # Generate the image based on the user's text input
        with st.spinner("Generating the image..."):
            image = pipeline(text_input, generator=generator).images[0]

        # Prepare the image for display and download
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        st.image(img_buffer, caption="Generated Image")
        st.download_button(
            "Download the image",
            img_buffer,
            "generated_image.jpg",
            "image/jpeg"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

