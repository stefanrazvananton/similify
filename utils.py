from flask import Flask, request, render_template, redirect, url_for
from flask_socketio import SocketIO, emit
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import base64
from torch.utils.data import DataLoader
import tqdm  # For progress display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors




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

class FeatureExtractorResNetContrastive(nn.Module):
    def __init__(self):
        super(FeatureExtractorResNetContrastive, self).__init__()
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        # Modify the last layer to output embeddings of size 256
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 256)

    def forward(self, x):
        output = self.model(x)
        return output

class FeatureExtractorVggContrastive(nn.Module):
    def __init__(self):
        super(FeatureExtractorVggContrastive, self).__init__()
        # Load pretrained VGG16
        self.model = models.vgg16(pretrained=True)
        # Modify the classifier to output embeddings of size 256
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 256)
    
    def forward(self, x):
        output = self.model(x)
        return output
