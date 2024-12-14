import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
import tqdm  # For progress display
import numpy as np
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
import torch.nn as nn
from torchvision import models

torch.backends.cudnn.benchmark = True

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])

# Custom Dataset for Siamese Network
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.classes = self.imageFolderDataset.classes
        self.class_to_idx = self.imageFolderDataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.image_paths = self.imageFolderDataset.imgs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get an image and its label
        img0_tuple = self.image_paths[index]
        img0_path, label0 = img0_tuple

        # Decide whether to make a positive or negative pair
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            # Keep looping till a different image of the same class is found
            while True:
                idx = random.randint(0, len(self.image_paths) - 1)
                img1_tuple = self.image_paths[idx]
                img1_path, label1 = img1_tuple
                if label0 == label1 and img0_path != img1_path:
                    break
        else:
            # Keep looping till an image of a different class is found
            while True:
                idx = random.randint(0, len(self.image_paths) - 1)
                img1_tuple = self.image_paths[idx]
                img1_path, label1 = img1_tuple
                if label0 != label1:
                    break

        # Load images
        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # Label is 0 if same class, 1 if different class
        label = torch.tensor([int(label0 != label1)], dtype=torch.float32)

        return img0, img1, label

# Path to your dataset
data_dir = 'imgs'  # Replace with your dataset path

# Create the dataset and dataloader
image_folder_dataset = datasets.ImageFolder(root=data_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=image_folder_dataset,
                                        transform=data_transforms)

train_loader = DataLoader(
    siamese_dataset,
    shuffle=True,
    batch_size=8,   # Increase batch size if GPU memory allows
)

# Define the Siamese Network using ResNet50
class SiameseNetwork_resnet50(nn.Module):
    def __init__(self):
        super(SiameseNetwork_resnet50, self).__init__()
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        # Modify the last layer to output embeddings of size 256
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 256)

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2):
        # Pass both inputs through the network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2






# Define the Siamese Network using VGG16
class SiameseNetwork_vgg16(nn.Module):
    def __init__(self):
        super(SiameseNetwork_vgg16, self).__init__()
        # Load pretrained VGG16
        self.model = models.vgg16(pretrained=True)
        # Modify the classifier to output embeddings of size 256
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 256)
    
    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2):
        # Pass both inputs through the network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute Euclidean distance between output1 and output2
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        # Compute the contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive



def extract_features_batch(dataloader, model):
    features_db = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, _ in tqdm.tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)  # Move batch of images to device
            batch_features = model.forward_once(images)  # Get features for batch
            batch_features = batch_features.cpu().numpy()  # Move to CPU and convert to numpy
            features_db.append(batch_features)

    # Stack all features into a single array
    return np.vstack(features_db)

def extract_features(image):
    with torch.no_grad():
        features = net.forward_once(image)
        features = features.cpu().numpy().flatten()  # Move to CPU and convert to numpy
        return features


def find_similar_images(query_image_path, features_db, top_n=10):
    

    
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = transform(query_image)
    query_image = query_image.to(device)  # Move query image to device
    query_features = extract_features(query_image.unsqueeze(0))  # Extract features for query image


    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(features_db)
    distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)
    distances = distances[0]
    indices = indices[0] 
    
    most_similar_indices = indices[:48]
    distances = distances[:48]

    similar_images = [(dataset.imgs[idx], 1 - distances[i]) for i, idx in enumerate(most_similar_indices)]
    
    return similar_images



transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])


dataset = datasets.ImageFolder(root='imgs', transform=transform)
batch_size = 16  # Adjust batch size based on GPU memory
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)




# Initialize the network, criterion, and optimizer
net = SiameseNetwork_resnet50()

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Load the saved state dictionary
net.load_state_dict(torch.load('siamese_resnet50_9.pth', map_location=device))

# Set the model to evaluation mode
net.eval()

features_db = extract_features_batch(data_loader, net)
np.save("features_resnet_contrastive_9.npy",features_db)





# Initialize the network, criterion, and optimizer
net = SiameseNetwork_vgg16()

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Load the saved state dictionary
net.load_state_dict(torch.load('siamese_vgg16_9.pth', map_location=device))

# Set the model to evaluation mode
net.eval()

features_db = extract_features_batch(data_loader, net)
np.save("features_vgg_contrastive_9.npy",features_db)
