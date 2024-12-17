import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

torch.backends.cudnn.benchmark = True

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
       
        img0_tuple = self.image_paths[index]
        img0_path, label0 = img0_tuple

        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                idx = random.randint(0, len(self.image_paths) - 1)
                img1_tuple = self.image_paths[idx]
                img1_path, label1 = img1_tuple
                if label0 == label1 and img0_path != img1_path:
                    break
        else:
            while True:
                idx = random.randint(0, len(self.image_paths) - 1)
                img1_tuple = self.image_paths[idx]
                img1_path, label1 = img1_tuple
                if label0 != label1:
                    break

        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = torch.tensor([int(label0 != label1)], dtype=torch.float32)

        return img0, img1, label


data_dir = 'imgs'

image_folder_dataset = datasets.ImageFolder(root=data_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=image_folder_dataset,
                                        transform=data_transforms)

train_loader = DataLoader(
    siamese_dataset,
    shuffle=True,
    batch_size=8,
)

class SiameseNetwork_resnet50(nn.Module):
    def __init__(self):
        super(SiameseNetwork_resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 256)

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2






class SiameseNetwork_vgg16(nn.Module):
    def __init__(self):
        super(SiameseNetwork_vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 256)
    
    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive



def extract_features_batch(dataloader, model):
    features_db = []
    model.eval() 
    with torch.no_grad(): 
        for images, _ in tqdm.tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device) 
            batch_features = model.forward_once(images) 
            batch_features = batch_features.cpu().numpy()
            features_db.append(batch_features)

    return np.vstack(features_db)

def extract_features(image):
    with torch.no_grad():
        features = net.forward_once(image)
        features = features.cpu().numpy().flatten()
        return features


def find_similar_images(query_image_path, features_db, top_n=10):
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = transform(query_image)
    query_image = query_image.to(device)  
    query_features = extract_features(query_image.unsqueeze(0)) 


    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(features_db)
    distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)
    distances = distances[0]
    indices = indices[0] 
    
    most_similar_indices = indices[:48]
    distances = distances[:48]

    similar_images = [(dataset.imgs[idx], 1 - distances[i]) for i, idx in enumerate(most_similar_indices)]
    
    return similar_images



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root='imgs', transform=transform)
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


net = SiameseNetwork_resnet50()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

net.load_state_dict(torch.load('siamese_resnet50_9.pth', map_location=device))

net.eval()

features_db = extract_features_batch(data_loader, net)
np.save("features_resnet_contrastive_9.npy",features_db)






net = SiameseNetwork_vgg16()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

net.load_state_dict(torch.load('siamese_vgg16_9.pth', map_location=device))

net.eval()

features_db = extract_features_batch(data_loader, net)
np.save("features_vgg_contrastive_9.npy",features_db)
