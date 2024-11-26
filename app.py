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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'secret!'  # Required for SocketIO

socketio = SocketIO(app)  # Initialize SocketIO

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
cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

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

# Function to find most similar images using selected similarity metric
def find_similar_images(query_features, features_db, metric):
    # Compute similarities/distances based on the selected metric
    if metric == 'cosine':
        similarities = cosine_similarity([query_features], features_db)[0]
        # Higher values mean more similar
        most_similar_indices = similarities.argsort()[-48:][::-1]
        # Retrieve most similar images and similarity scores
        similar_images = [(cifar10_dataset.data[idx], similarities[idx]) for idx in most_similar_indices]
    else:
        # For distance metrics, lower values mean more similar
        if metric == 'euclidean':
            distances = cdist([query_features], features_db, metric='euclidean')[0]
        elif metric == 'manhattan':
            distances = cdist([query_features], features_db, metric='cityblock')[0]
        else:
            # Default to Euclidean if invalid metric
            distances = cdist([query_features], features_db, metric='euclidean')[0]
        # Get indices of most similar images
        most_similar_indices = distances.argsort()[:48]
        # Retrieve most similar images and distance scores
        similar_images = [(cifar10_dataset.data[idx], distances[idx]) for idx in most_similar_indices]

    return similar_images

# Updated extract_features function to add noise (same as before)
def extract_features(image, model, mu, sigma):
    with torch.no_grad():
        features = model(image)
        features = features.cpu().numpy().flatten()  # Move to CPU and convert to numpy
        # Add Gaussian noise
        noise = np.random.normal(mu, sigma, len(features))
        return features + noise

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded and required fields are present
        if 'file' not in request.files or 'model' not in request.form or 'metric' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        model_name = request.form['model']
        metric = request.form['metric']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Redirect to the display route with the selected parameters
            return redirect(url_for('display_results', filename=file.filename, model=model_name, metric=metric))
    return render_template('upload.html')

@app.route('/results/<filename>')
def display_results(filename):
    model_name = request.args.get('model', 'resnet50')
    metric = request.args.get('metric', 'cosine')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Select the appropriate model and features_db
    if model_name == 'resnet50':
        model = resnet_model
        features_db = features_resnet50
    elif model_name == 'vgg16':
        model = vgg_model
        features_db = features_vgg16
    else:
        # Invalid model selected, default to ResNet50
        model = resnet_model
        features_db = features_resnet50

    # Precompute features for the uploaded image without noise
    query_image = Image.open(filepath).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze(0).to(device)
    base_query_features = extract_features(query_image_tensor, model, mu=0, sigma=0)

    # Initial similar images with default mu and sigma
    similar_images = find_similar_images(base_query_features, features_db, metric)
    # Convert images to base64 to display on the webpage
    images = []
    for img_array, similarity in similar_images:
        img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images.append({'img_data': img_str, 'similarity': f"{similarity:.4f}"})
    
    # Convert the uploaded image to base64
    with open(filepath, "rb") as f:
        uploaded_image_data = base64.b64encode(f.read()).decode()

    return render_template('results.html', images=images, uploaded_image=uploaded_image_data, model_name=model_name, metric=metric, filename=filename)

# SocketIO event handlers
@socketio.on('update_parameters')
def handle_update_parameters(data):
    mu = float(data['mu'])
    sigma = float(data['sigma'])
    metric = data['metric']
    model_name = data['model']
    filename = data['filename']

    # Select the appropriate model and features_db
    if model_name == 'resnet50':
        model = resnet_model
        features_db = features_resnet50
    elif model_name == 'vgg16':
        model = vgg_model
        features_db = features_vgg16
    else:
        # Invalid model selected, default to ResNet50
        model = resnet_model
        features_db = features_resnet50

    # Load the query image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    query_image = Image.open(filepath).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze(0).to(device)

    # Extract features with updated mu and sigma
    query_features = extract_features(query_image_tensor, model, mu, sigma)

    # Find similar images
    similar_images = find_similar_images(query_features, features_db, metric)

    # Convert images to base64
    images = []
    for img_array, similarity in similar_images:
        img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images.append({'img_data': img_str, 'similarity': f"{similarity:.4f}"})

    # Send the updated images back to the client
    emit('update_images', {'images': images})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)