import torch
from torchvision import datasets, transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

from test import Net


############################# CHANGE THIS ###############################

model_path = 'mnist_cnn.pt'  # Replace with the path to your model file

############################# CHANGE THIS ###############################



# Load the pre-trained model from a .pt file
def load_model(model_path):
    model = Net()  # Instantiate your model architecture
    model.load_state_dict(torch.load(model_path))  # Load the weights
    model.eval()  # Set the model to evaluation mode
    return model


# Function to classify a single image and display the result
def classify_and_show(model, image, label):
    # Add batch dimension and pass the image to the model
    with torch.no_grad():
        image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
        output = model(image)       # Forward pass through the model
        _, predicted = torch.max(output, 1)  # Get the predicted class
    
    # Convert the Tensor image back to a PIL image for display
    image_pil = transforms.ToPILImage()(image.squeeze(0))  # Remove batch dimension

    # Display the image and predicted class
    plt.imshow(image_pil, cmap='gray')
    plt.title(f'Predicted: {predicted.item()}, Actual: {label}')
    plt.show()

# Define the transformation for the dataset
transform = transforms.ToTensor()

# Download the MNIST dataset
dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

# Load the pre-trained model
model = load_model(model_path)

# Classify and display the first 5 images in the dataset
for i in range(5):
    image, label = dataset[i]  # Get the image and label
    classify_and_show(model, image, label)
