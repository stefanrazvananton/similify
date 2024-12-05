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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Transformation applied to all images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])


# Load CIFAR-10 dataset
cifar10_dataset = datasets.ImageFolder(root='imgs', transform=transform)



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

# Load both models and their feature databases
# ResNet50 model and features
resnet_model = FeatureExtractorResNet()
resnet_model = resnet_model.to(device)
resnet_model.eval()
features_resnet50 = np.load('features_resnet50.npy')

# VGG16 model and features
vgg_model = FeatureExtractorVGG()
vgg_model = vgg_model.to(device)
vgg_model.eval()
features_vgg16 = np.load('features_vgg16.npy')

# ResNet50 contrastive model and features
resnet_contrastive_model = FeatureExtractorResNetContrastive()
resnet_contrastive_model = resnet_contrastive_model.to(device)
resnet_contrastive_model.eval()
resnet_contrastive_model.load_state_dict(torch.load('siamese_resnet50.pth', map_location=device))
features_resnet_contrastive = np.load('features_resnet50_contrastive.npy')

# VGG16 contrastive model and features
vgg_contrastive_model = FeatureExtractorVggContrastive()
vgg_contrastive_model = vgg_contrastive_model.to(device)
vgg_contrastive_model.eval()
vgg_contrastive_model.load_state_dict(torch.load('siamese_vgg16_3.pth', map_location=device))
features_vgg_contrastive = np.load('features_vgg_contrastive.npy')


# Function to find most similar images using selected similarity metric
def find_similar_images(query_features, features_db, metric):
    # Compute similarities/distances based on the selected metric
    if metric == 'cosine':

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(features_db)
        distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)
        
        distances = distances[0]
        indices = indices[0] 

        similar_images = [(cifar10_dataset.imgs[idx], 1 - distances[i]) for i, idx in enumerate(indices)]
    else:
        # For distance metrics, lower values mean more similar
        if metric == 'euclidean':
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean').fit(features_db)
            distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)
        elif metric == 'manhattan':
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='manhattan').fit(features_db)
            distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)


        distances = distances[0]
        indices = indices[0] 
        
        similar_images = [(cifar10_dataset.imgs[idx], distances[i]) for i, idx in enumerate(indices)]

    return similar_images



# Updated extract_features 
def extract_features(image, model):
    with torch.no_grad():
        features = model(image)
        features = features.cpu().numpy().flatten()  # Move to CPU and convert to numpy

        return features


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded and required fields are present
        if 'file' not in request.files or 'metric' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        metric = request.form['metric']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Redirect to the display route with the selected parameters
            return redirect(url_for('display_results', filename=file.filename, metric=metric))
    return render_template('upload.html')



@app.route('/results/<filename>')
def display_results(filename):
    metric = request.args.get('metric', 'cosine')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load the query image
    query_image = Image.open(filepath).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze(0).to(device)

    # Extract features using both models
    query_features_resnet = extract_features(query_image_tensor, resnet_model)
    query_features_vgg = extract_features(query_image_tensor, vgg_model)
    query_features_resnet_contrastive = extract_features(query_image_tensor, resnet_contrastive_model)
    query_features_vgg_contrastive = extract_features(query_image_tensor, vgg_contrastive_model)

    
    # Find similar images for both models
    similar_images_resnet = find_similar_images(query_features_resnet, features_resnet50, metric)
    similar_images_vgg = find_similar_images(query_features_vgg, features_vgg16, metric)
    similar_images_resnet_contrastive = find_similar_images(query_features_resnet_contrastive, features_resnet_contrastive, metric)
    similar_images_vgg_contrastive = find_similar_images(query_features_vgg_contrastive, features_vgg_contrastive, metric)
    
    # Convert images to base64 and include their filenames
    images_resnet = []
    for img_array, similarity in similar_images_resnet:
        img = Image.open(img_array[0])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_resnet.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array[0])  # Include file name
        })

    images_vgg = []
    for img_array, similarity in similar_images_vgg:
        img = Image.open(img_array[0])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_vgg.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array[0])  # Include file name
        })

    images_resnet_contrastive = []
    for img_array, similarity in similar_images_resnet_contrastive:
        img = Image.open(img_array[0])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_resnet_contrastive.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array[0])  # Include file name
        })

    images_vgg_contrastive = []
    for img_array, similarity in similar_images_vgg_contrastive:
        img = Image.open(img_array[0])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_vgg_contrastive.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array[0])  # Include file name
        })

    

    # Convert the uploaded image to base64
    with open(filepath, "rb") as f:
        uploaded_image_data = base64.b64encode(f.read()).decode()

    return render_template('results.html', images_resnet=images_resnet, images_vgg=images_vgg,images_resnet_contrastive=images_resnet_contrastive,images_vgg_contrastive=images_vgg_contrastive,
                           uploaded_image=uploaded_image_data, metric=metric, filename=filename)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)