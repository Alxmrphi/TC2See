import re

import numpy as np
import torch
import clip
from PIL import Image
import glob
from pathlib import Path
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

embeddings = {}
#for filename in sorted(glob.glob("/home/richard/PycharmProjects/bird_data/stim3/images/*.jpg")):
for filename in sorted(glob.glob("/home/richard/PycharmProjects/bird_data/docs/cropped/*.png")):

    if "hash" in filename:
        continue

    #name = re.match(r"(.*)_\d*.", Path(filename).stem).groups()[0]
    name = Path(filename).stem
    print(name)

    image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features = np.array(image_features)
    print(name, image_features.shape)
    embeddings[name] = image_features[0]

np.save("embeddings_bird", embeddings)
