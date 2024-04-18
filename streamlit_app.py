import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import io

st.title("Générateur prompt to Image - Logo")
text_input = st.text_input("Entrez le prompt que vous souhaitez:", "")

if text_input:
    try:
        # Ensures compatibility with your environment (e.g., checks if CUDA is available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = AutoPipelineForText2Image.from_pretrained("logo-wizard/logo-diffusion-checkpoint", torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device).manual_seed(31)

        with st.spinner("Génération de l'image en cours..."):
            image = pipeline(text_input, generator=generator).images[0]

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        st.image(img_buffer, caption="Image générée")
        st.download_button(
            label="Télécharger l'image",
            data=img_buffer,
            file_name="image_generee.jpg",
            mime="image/jpeg"
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
