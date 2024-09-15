
import streamlit as st
import faiss
import torch
import clip
import numpy as np
from PIL import Image

# Load the model and device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the FAISS search class (a simplified version based on the notebook content)
class Myfaiss:
    def __init__(self, index_path, id2img_fps, device, model):
        self.device = device
        self.model = model
        self.index = faiss.read_index(index_path)
        self.id2img_fps = id2img_fps
    
    def text_search(self, text, k):
        text = clip.tokenize([text]).to(self.device)
        text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        image_paths = list(map(self.id2img_fps.get, idx_image))
        return scores, image_paths
    
    def show_images(self, image_paths):
        images = [Image.open(img_path) for img_path in image_paths]
        return images

# Load the FAISS index and the id2img_fps dictionary
bin_file = 'faiss_normal_ViT.bin'
id2img_fps = {}  # Load the mapping from image IDs to file paths (this will need to be provided)

# Initialize FAISS
faiss_test = Myfaiss(bin_file, id2img_fps, device, model)

# Streamlit UI
st.title("FAISS-CLIP Image Search App")

# Text input
text_input = st.text_input("Enter search text:", "")

# Set the number of results to display
num_results = st.slider("Number of results to display:", 1, 100, 10)

if st.button("Search"):
    if text_input:
        scores, image_paths = faiss_test.text_search(text_input, k=num_results)
        images = faiss_test.show_images(image_paths)
        
        # Display images
        for img in images:
            st.image(img)
    else:
        st.write("Please enter some text to search for.")
