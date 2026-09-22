# CodeAlpha Handwritten Character Recognition

## Project Overview

This project is a Handwritten Character Recognition system developed using Python, TensorFlow and Convolutional Neural Network (CNN).

The model is trained using the MNIST handwritten digit dataset and recognizes handwritten digits from 0 to 9.

## Technologies Used

- Python
- TensorFlow
- NumPy
- CNN
- MNIST Dataset

## Dataset

- 60,000 training images
- 10,000 testing images
- Image size: 28 x 28 pixels
- 10 classes: digits 0 to 9

## Model Architecture

- Convolutional Layer
- Max Pooling Layer
- Flatten Layer
- Dense Layer
- Output Layer with Softmax

## Results

The CNN model achieved a test accuracy of approximately 98.56%.

## Project Files

- `main.py` - Main Python program
- `handwritten_character_model.keras` - Trained CNN model
- `README.md` - Project documentation

## How to Run

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib