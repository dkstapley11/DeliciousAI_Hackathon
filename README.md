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


# Process/Implementation - Danny

- **Create Custom Datasets** - Inheriting from the Dataset class in the PyTorch library, I made two subclasses: CustomTrainDataset and CustomTestDataset. I did this because we were going to be treating our training and testing datasets differently. Furthermore, the training data was given in a text file as paths to each image from the `/images` directory, followed by its associated class. The testing data was just a path to each image.
```python
# Custom dataset for training
class CustomTrainDataset(Dataset):
    def __init__(self, annotations_file, img_dir, class_yaml, transform=None):
        self.img_labels = [line.strip().split(',') for line in open(annotations_file)]
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_index = load_class_mapping(class_yaml) 

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label_str = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.label_to_index[label_str]
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom dataset for testing
class CustomTestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_paths = [line.strip() for line in open(annotations_file)]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_paths[idx] 
```