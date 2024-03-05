import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0
import torch.nn as nn
import torch.optim as optim

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = datasets.ImageFolder(root='/Users/c-mac/Research/ML Practice/Delicious Hackathon/bev_classification/images', transform=transform)

# Load train labels
with open('/Users/c-mac/Research/ML Practice/Delicious Hackathon/bev_classification/datasets/train.txt', 'r') as file:
    train_labels = file.read().splitlines()

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Define DataLoader for training and validation sets
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Load the pre-trained EfficientNet model
model = efficientnet_b0(pretrained=True)

# Update the number of classes based on your requirements
num_classes = 99
model.fc = torch.nn.Linear(model._fc.in_features, num_classes)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
