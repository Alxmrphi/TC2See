import os
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
import h5py
from tqdm import tqdm

def apply_occlusion(image, mask_size=67, stride=33):
    """Apply a sliding window occlusion to an image. (bird images are 335x335)"""
    H, W, C = image.shape
    occluded_images = []
    
    for y in range(0, H - mask_size + 1, stride):
        for x in range(0, W - mask_size + 1, stride):
            occluded_img = image.copy()
            occluded_img[y:y+mask_size, x:x+mask_size] = (0, 0, 0)
            occluded_images.append(occluded_img)
    
    return occluded_images

def save_embeddings(embeddings, output_path):
    """Save embeddings as .npy files."""
    np.save(output_path, embeddings)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    model, preprocess = clip.load("ViT-B/32", device=device)

    input_folder = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/cropped")
    output_folder = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occluded_embeddings/size_67")
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_folder.glob("*.png"))[0:10]

    for image_path in tqdm(image_paths, desc="Processing Images"):
        image_name = image_path.stem
        img = np.array(Image.open(image_path).convert("RGB"))
        occluded_images = apply_occlusion(img)

        image_output_dir = output_folder / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        embeddings = []
        for i, occluded_img in enumerate(occluded_images):
            pil_img = Image.fromarray(occluded_img)
            processed_img = preprocess(pil_img).unsqueeze(0).to(device)  # Move to GPU

            with torch.no_grad():
                image_features = model.encode_image(processed_img).to(device)  # Ensure tensor is on GPU

            embedding = image_features.cpu().numpy().squeeze()  # Move back to CPU for saving
            embeddings.append(embedding)

            save_embeddings(embedding, image_output_dir / f"occlusion_{i}.npy")

    print("Processing complete.")

    
if __name__ == "__main__":
    main()