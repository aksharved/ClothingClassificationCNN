# ClothingClassificationCNN

This repository contains a Convolutional Neural Network (CNN) implementation for classifying images in the FashionMNIST dataset using PyTorch. It has an accuracy of 92.42%. The project includes data augmentation, batch normalization, and dropout for improved performance and generalization. It provides scripts for training, testing, and visualizing the model's performance.

## Project Structure

```plaintext
ClothingClassificationCNN/
├── models/                          # Directory to save and load models
├── src/
│   ├── model.py                     # Definition of the CNN model
│   ├── plotting.py                  # Script for plotting training and validation losses
│   ├── test.py                      # Script for testing the model
│   ├── train.py                     # Script for training the model
├── README.md                        # Readme file for the project
