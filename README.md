# PyTorch Card Recognizer

This is an introductory project uses **PyTorch** and **EfficientNet** to classify playing cards from images.  
It is designed to demonstrate end-to-end training and inference of an image classifier, with clean architecture and custom dataset support.

---

## Description

This repository includes:
- A **training pipeline** with EfficientNet-B0 using `timm`
- A **custom dataset class** using `ImageFolder`
- **MPS (Metal)** support for fast training on macOS
- Visualization of **training and validation loss**
- A script for **inference** and prediction confidence visualization

---

## Model Overview

The model architecture uses:
- **EfficientNet B0** as a base feature extractor
- A custom **linear classifier head**
- **53 output classes**, including all playing cards and jokers

---

## Data Visualization

Below are the graphs from training the model and asking it to identify the cards' identities in the testing dataset.
The first one is from `cardrecognition.py`, the other three are from `predictcard.py`. Predictions 1 nd 2 are sourced from the provided `cards_dataset` while the third prediction was a randomly selected image from the internet.

![Loss Over Epoch](loss_over_epoch.png)
![Sample Prediction 1](sampleprediction1.png)
![Sample Prediction 2](sampleprediction2.png)
![Sample Prediction 3](sampleprediction3.png)
