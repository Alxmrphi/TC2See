import os
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt  
import pandas as pd
from tqdm import tqdm
import json

patch_size = 32

# Set paths
input_folder = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/cropped")
embedding_root = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occluded_embeddings/size_{patch_size}")
original_embeddings_path = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/original_embeddings.npy")
output_plot_dir = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occlusion_violin_plots/violins_size_{patch_size}")
output_plot_dir.mkdir(exist_ok=True)

# Load CLIP model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Compute and save original embeddings
original_embeddings = {}
image_paths = sorted(input_folder.glob("*.png"))

for image_path in tqdm(image_paths, desc="Processing Original Images"):
    image_name = image_path.stem
    pil_img = Image.open(image_path).convert("RGB")
    processed_img = preprocess(pil_img).unsqueeze(0).to(device)  # Move image to GPU
    
    with torch.no_grad():
        image_features = model.encode_image(processed_img)  # Run inference on GPU
    
    original_embeddings[image_name] = image_features.cpu().numpy().squeeze()  # Move back to CPU for storage

image_metrics_list = []

# Loop over images
for image_folder in tqdm(embedding_root.iterdir(), desc="Processing Occluded Embeddings"):
    if not image_folder.is_dir():
        continue
    
    image_name = image_folder.name
    original_embedding = original_embeddings.get(image_name)
    
    if original_embedding is None:
        print(f"Skipping {image_name} (no original embedding found)")
        continue
    
    # Prepare distance data
    distance_data = []
    
    # Load occluded embeddings
    for occlusion_file in image_folder.glob("occlusion_*.npy"):
        occluded_embedding = np.load(occlusion_file)

        # Compute distances (remains on CPU)
        cosine_distance = 1 - np.dot(original_embedding, occluded_embedding) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(occluded_embedding)
        )

        # Store results
        distance_data.append({"Distance Type": "Cosine", "occlusion_id": int(occlusion_file.stem.split("_")[1]),  "Distance": cosine_distance})
    
    # Convert to DataFrame
    df = pd.DataFrame(distance_data)

    mean_cosine_distance = float(df["Distance"].mean())
    std_cosine_distance = float(df["Distance"].std())  
    max_cosine_distance = df.loc[df["Distance"].idxmax()]
    # Get the occlusion_id of the max cosine distance
    max_cosine_distance_id = float(max_cosine_distance["occlusion_id"])

    image_metrics_list.append({
        "image_name": image_name, 
        "mean_cosine_distance": mean_cosine_distance, 
        "std_cosine_distance": std_cosine_distance,
        "max_cosine_distance_id": max_cosine_distance_id
    })

    # Plot violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Distance Type", y="Distance", data=df, inner="point", scale="width")
    plt.title(f"Occlusion Distance Distribution: {image_name}  ({patch_size}x{patch_size} occlusions)")
    plt.ylabel(f"Distance (mean distance: {mean_cosine_distance:.2f})")
    plt.xlabel("Distance Metric")
    
    # Save plot
    output_plot_path = output_plot_dir / f"{image_name}_violin_plot.png"
    plt.savefig(output_plot_path)
    plt.close()
    
    print(f"Violin plot saved to {output_plot_path}")

# Save metrics list to json
with open(output_plot_dir / f"occ_{patch_size}x_{patch_size}_image_embeddings_stats.json", "w") as f:
    json.dump(image_metrics_list, f, indent=4)
