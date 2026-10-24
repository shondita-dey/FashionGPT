import clip
import torch
import faiss
import pickle
import numpy as np
from PIL import Image

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD CLIP MODEL
# --------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)

# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------
index = faiss.read_index(
    "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\fashiongpt\\fashion_faiss.index"
)

with open(
    "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\fashiongpt\\image_paths.pkl",
    "rb",
) as f:
    image_paths = pickle.load(f)


# --------------------------------------------------
# TEXT EMBEDDING
# --------------------------------------------------
def embed_text(text):

    tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().astype("float32")


# --------------------------------------------------
# IMAGE EMBEDDING
# --------------------------------------------------
def embed_image(image):

    # If uploaded file passed instead of PIL image
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().astype("float32")


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------
def search_images(query_embedding, top_k=5):

    D, I = index.search(query_embedding, top_k)

    results = [image_paths[i] for i in I[0]]
    scores = D[0]

    return results, scores