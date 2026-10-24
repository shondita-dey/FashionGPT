import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\data")
DEEPFASHION_DIR = os.path.join(DATA_DIR, "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\data\\DeepFashion2 Dataset")
FASHIONREC_DIR = os.path.join(DATA_DIR, "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\data\\item_images")
POLYVORE_DIR = os.path.join(DATA_DIR, "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\data\\Re-PolyVore")


import faiss
import torch
import clip
import pickle
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def get_all_images(root_folder):
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


all_images = []
all_images += get_all_images(DEEPFASHION_DIR)
all_images += get_all_images(FASHIONREC_DIR)
all_images += get_all_images(POLYVORE_DIR)

print("Total images found:", len(all_images))

embeddings = []

batch_size = 32
for i in tqdm(range(0, len(all_images), batch_size)):
    batch_paths = all_images[i:i+batch_size]
    images = []

    for path in batch_paths:
        try:
            img = preprocess(Image.open(path).convert("RGB"))
            images.append(img)
        except:
            continue

    if not images:
        continue

    images = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    embeddings.append(image_features.cpu())

embeddings = torch.cat(embeddings).numpy()

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\fashiongpt\\fashion_faiss.index")

with open("D:\\M.Sc. Data Analytics\\SY\\Semester 4\\Project\\fashiongpt\\image_paths.pkl", "wb") as f:
    pickle.dump(all_images, f)

print("FAISS index built successfully!")