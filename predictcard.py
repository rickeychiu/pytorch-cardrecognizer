import torch
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import os
from cardrecognition import simpleCardClassifer, PlayingCardDataset

import matplotlib
matplotlib.use('Agg')  # Ensures a clean pop-up window
import matplotlib.pyplot as plt
plt.close('all')  # Clears any lingering figures

# ========== CONFIG ==========
NUM_CLASSES = 53
MODEL_PATH = "trained_model.pth"  # Update if saved under a different name
IMAGE_PATH = "/Users/rickeychiu/Desktop/Personal Coding/pytorch-stuff/cards_dataset/valid/ten of clubs/5.jpg"  # Replace with your actual test image
DATASET_DIR = "/Users/rickeychiu/Desktop/Personal Coding/pytorch-stuff/cards_dataset/train"  # To load class names
IMAGE_SIZE = (128, 128)
# ============================

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = simpleCardClassifer(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load class names
dataset = PlayingCardDataset(DATASET_DIR, transform=transforms.ToTensor())
class_names = dataset.classes

# Preprocess test image
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

original_image, image_tensor = preprocess_image(IMAGE_PATH, transform)

# Predict
with torch.no_grad():
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()

# Visualize
def visualize_predictions(image, probs, class_names, top_k=5):
    top_indices = probs.argsort()[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_classes = [class_names[i] for i in top_indices]

    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    axarr[0].imshow(image)
    axarr[0].axis("off")
    axarr[0].set_title("Input Image")

    axarr[1].barh(top_classes[::-1], top_probs[::-1])
    axarr[1].set_xlim(0, 1)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Top Predictions")

    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    print("Saved prediction visualization to prediction_plot.png")

visualize_predictions(original_image, probabilities, class_names)
