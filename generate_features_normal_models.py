import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tqdm  # For progress display
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose architecture: 'resnet50' or 'vgg16'


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])


dataset = datasets.ImageFolder(root='imgs', transform=transform)

# Step 2: Load pretrained model and modify it to remove the final layer

# Define feature extractor classes
class FeatureExtractorResNet(nn.Module):
    def __init__(self):
        super(FeatureExtractorResNet, self).__init__()
        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])  # Exclude the last FC layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 2048)
        return x

class FeatureExtractorVGG(nn.Module):
    def __init__(self):
        super(FeatureExtractorVGG, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool  # AdaptiveAvgPool2d
        # Include the classifier layers up to the second last layer
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
architecture = 'vgg16'  # or 'vgg16'
# Instantiate the model based on the architecture
if architecture == 'resnet50':
    model = FeatureExtractorResNet()
elif architecture == 'vgg16':
    model = FeatureExtractorVGG()

model = model.to(device)
model.eval()

# Step 3: Define DataLoader with batch processing
batch_size = 64  # Adjust batch size based on GPU memory
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Batch feature extraction function
def extract_features_batch(dataloader, model):
    features_db = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in tqdm.tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)  # Move batch of images to device
            batch_features = model(images)  # Get features for batch
            batch_features = batch_features.cpu().numpy()  # Move to CPU and convert to numpy
            features_db.append(batch_features)

    # Stack all features into a single array
    return np.vstack(features_db)


features_db = extract_features_batch(data_loader, model)
np.save("features_vgg16.npy",features_db)




architecture = 'resnet50'  # or 'vgg16'
# Instantiate the model based on the architecture
if architecture == 'resnet50':
    model = FeatureExtractorResNet()
elif architecture == 'vgg16':
    model = FeatureExtractorVGG()

model = model.to(device)
model.eval()

# Step 3: Define DataLoader with batch processing
batch_size = 64  # Adjust batch size based on GPU memory
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Batch feature extraction function
def extract_features_batch(dataloader, model):
    features_db = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in tqdm.tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)  # Move batch of images to device
            batch_features = model(images)  # Get features for batch
            batch_features = batch_features.cpu().numpy()  # Move to CPU and convert to numpy
            features_db.append(batch_features)

    # Stack all features into a single array
    return np.vstack(features_db)


features_db = extract_features_batch(data_loader, model)
np.save("features_resnet50.npy",features_db)

