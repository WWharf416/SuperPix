import torch
import pyiqa
from PIL import Image
import os
from pathlib import Path

def compute_niqe_scores(folder_path):
    # Create NIQE model
    model = pyiqa.create_metric('niqe')
    
    # Get all image files in the folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    folder_path = Path(folder_path)
    
    # Process each image in the folder
    for img_path in folder_path.glob('*'):
        if img_path.suffix.lower() in image_extensions:
            try:
                # Load and convert image
                img = Image.open(img_path).convert('RGB')
                # Compute NIQE score
                score = model(img)
                print(f"Image: {img_path.name}, NIQE score: {score.item():.4f}")
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    folder_path = input("Enter the path to the folder containing images: ")
    compute_niqe_scores(folder_path)