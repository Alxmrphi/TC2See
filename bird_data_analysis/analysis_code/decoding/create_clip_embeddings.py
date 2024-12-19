import re
import numpy as np
import torch
import clip
from PIL import Image
import glob
from pathlib import Path
import re
import h5py

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

embeddings = []
for filename in sorted(glob.glob("../../data/masked_images/*.png")):

    if "hash" in filename:
        continue

    name = Path(filename).stem
    print(name)

    image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features = image_features.cpu().numpy()
    embeddings.append(image_features[0]) 

embeddings_matrix = np.stack(embeddings, axis=0)

model_name = 'ViT-B=32'
embedding_name = 'embedding'

hdf5_file_path = f'../../data/{model_name}-features_bg_mask.hdf5'

with h5py.File(hdf5_file_path, 'w') as f:
    f.create_dataset(embedding_name, data=embeddings_matrix)

print(f"Embeddings saved to {hdf5_file_path}")