# DeliciousAI_Hackathon
March 2024 Artificial Intelligence Association hackathon competition. (Group: Daniel Stapley, Kai Sandberg, Aaron Earl, Collin McGregor, Jonathan Utley)

## Project Background

### What Delicious AI Does
- **Image Classification** - They use "computer vision, 3D point clouds, and sophisticated machine learning models to capture better data and deliver better intelligence in milliseconds." 
- **Who Uses Their Services** - Their principal goal is to help retailers use artificial intelligence to know when they need to restock. They assist in the optimization of these processes in order to maximize efficiency.

### The Problem
- **Limited Data** - Delicious AI has recruited the help of the BYU AI Association to classify images into a large number of classes. Comparative to the amount of classes, the dataset is relatively small, and therefore more difficult to classisfy with reliable accuracy.

## Project Objective
- **Sort Data** - The main goal of this project is to sort the provided dataset of images into classes with the highest degree of accuracy possible. Assign each classificaton a number between 1 and 5, denoting predicted degree of accuracy. 
- **Using** - Prioritizing accuracy above all else, we will use pre-trained machine learning models found in TorchVision library of PyTorch.

## General Project Plan
- **Prepare Data** - Organize our dataset into proper folders (train, validation, test), use 'torchvision.transforms' function within the PyTorch library to normalize images. Create a PyTorch DataLoader for each set
- **Prepare EfficientNet** - Load the pre-trained EfficientNet model from torchvision

```python
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, random_split

model = efficientnet_b0(pretrained=True)
```

- **Modify Output Layer** - Modify the output layer to match the number of classes in the dataset

```python
num_classes = pass
model.fc = torch.nn.Linear(model._fc.in_features, num_classes)
```

- **Define Loss Function and Optimizer** - Define functions to help train our model on our dataset

```python
criterion = torch.nn.CrossEntropyLoss()
Optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

- **Train Our Model**
```python
num_epochs = 10
for epoch in range(num_epochs):
  for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

- **Evaluate Performance** - After training, evaluate the model's performance with the validation set

```python
model.eval()
with torch.no_grad():
  for inputs, labels in val_loader:
    outputs = model(inputs)
```
- **Test the Model** - Finally, test the trained model on our test set and see how well it generalizes/classifies

```python
model.eval()
with torch.no_grad():
  for inputs, labels in test_loader:
    outputs = model(inputs)
```

- **Adjust Hyperparameters** - If necessary, try adjusting hyperparameters (like learning rate, batch size) and consider data augmentation techniques to improve performance


