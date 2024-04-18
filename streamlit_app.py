import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import io
import configparser

# Charger les configurations à partir du fichier config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Essayer d'obtenir le token API depuis le fichier de configuration
hf_token = config.get('DEFAULT', 'HUGGINGFACE_TOKEN', fallback=None)
if not hf_token:
    st.error("S'il vous plaît configurez votre token API dans le fichier 'config.ini' sous la section [DEFAULT] avec la clé HUGGINGFACE_TOKEN.")
    st.stop()

st.title("Générateur d'Images à partir de Texte - Utilisation des Modèles Hugging Face")
text_input = st.text_input("Entrez le prompt que vous souhaitez transformer en image:", "")

if text_input:
    try:
        # Vérifie si CUDA est disponible et l'utilise sinon utilise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Charge le modèle en utilisant le token pour l'authentification
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "logo-wizard/logo-diffusion-checkpoint", 
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to(device)
        
        # Définit un générateur de nombres aléatoires pour la reproductibilité
        generator = torch.Generator(device).manual_seed(42)

        # Génère l'image
        with st.spinner("Génération de l'image en cours..."):
            image = pipeline(text_input, generator=generator).images[0]

        # Prépare l'image pour l'affichage et le téléchargement
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        st.image(img_buffer, caption="Image générée")
        st.download_button(
            "Télécharger l'image",
            img_buffer,
            "image_generee.jpg",
            "image/jpeg"
        )

    except Exception as e:
        st.error(f"Une erreur est survenue: {str(e)}")

