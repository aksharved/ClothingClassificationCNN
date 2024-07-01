# Import necessary libraries
import torch
from model import ConvNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model weights
model_path = '../models/convnet_model.pth'
model = ConvNet().to(device)
model.load_state_dict(torch.load(model_path))
print(f'Model weights loaded from {model_path}')

# Datasets and DataLoaders
test_dataset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Testing function
def test(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of class {i}: {acc} %')

# Testing the model
test(model, test_loader, device)


