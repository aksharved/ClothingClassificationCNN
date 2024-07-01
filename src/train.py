# Import necessary libraries
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import ConvNet
from utils import plot_loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
epochs = 50
batch_size = 64
learning_rate = 0.001
model_path = '../models/convnet_model.pth'  # Path to save the model

# Data transformations with augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),      # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),              # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
])

# Datasets and DataLoaders
train_dataset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, optimizer, and scheduler
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()                      # Cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Learning rate scheduler

# Training function
def train(model, train_loader, criterion, optimizer, scheduler, device, epochs, model_path):
    model.train()  # Set model to training mode
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        scheduler.step()  # Update learning rate
        average_loss = epoch_loss / len(train_loader)
        losses.append(average_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

    # Save the trained model weights
    torch.save(model.state_dict(), model_path)
    print(f'Model weights saved to {model_path}')

    return losses

# Training the model
losses = train(model, train_loader, criterion, optimizer, scheduler, device, epochs, model_path)
plot_loss(losses)
