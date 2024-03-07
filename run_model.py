import torch
from torchvision import transforms, models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import yaml


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_class_mapping(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return {str(cls): idx for idx, cls in enumerate(yaml_content['classes'])}

class_yaml=os.path.expanduser('~/downloads/bev_classification/names.yaml')

def invert_mapping(mapping):
    return {v: k for k, v in mapping.items()}

class_mapping = load_class_mapping(class_yaml)
index_to_class = invert_mapping(class_mapping)

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
     
def main():
    test_dataset = CustomTestDataset(
        annotations_file=os.path.expanduser('~/downloads/bev_classification/datasets/test_edited.txt'),
        img_dir=os.path.expanduser('~/downloads/bev_classification/'),
        transform=transform
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)

    predictions = []  # A list to hold path and predictions

    # Initialize the model
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Update the number of classes in the classifier, if your dataset doesn't have 1000 classes
    num_classes = 99  # Update this based on your dataset
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_path = 'model_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for images, paths in test_loader:  # Now we also get paths
            images = images.to(device)
            outputs = model(images)

            _, top1_pred = outputs.topk(1, 1, True, True)
            _, top5_pred = outputs.topk(5, 1, True, True)
            
            top1_pred = top1_pred.squeeze().tolist()
            top5_pred = top5_pred.tolist()
            
            for i, path in enumerate(paths):
            # Convert top-1 and top-5 indices to class names
                top1_class_name = index_to_class[top1_pred[i] if type(top1_pred) is list else top1_pred]
                top5_class_names = [index_to_class[idx] for idx in top5_pred[i]]
            
                predictions.append((path, top1_class_name, *top5_class_names))

    # Save the top-1 predictions with class names
    with open('formatted_top1_predictions.txt', 'w') as f:
        for path, top1_class_name, *_ in predictions:
            f.write(f'{path}, {top1_class_name}\n')

    # Save the top-5 predictions with class names
    with open('formatted_top5_predictions.txt', 'w') as f:
        for path, _, *top5_class_names in predictions:
            f.write(f'{path}, ' + ', '.join(top5_class_names) + '\n')

if __name__ == '__main__':
    main()