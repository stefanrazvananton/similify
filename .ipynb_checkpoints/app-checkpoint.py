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
from utils import FeatureExtractorResNet,FeatureExtractorVGG,FeatureExtractorResNetContrastive,FeatureExtractorVggContrastive
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




import numpy as np
from PIL import Image
import networkx as nx



def build_similarity_graph(query_image_path, depth, N,metric,model):
    G = nx.Graph()
    visited = set()
    queue = [(query_image_path, 0)]  # (image_path, current_depth)
    while queue:
        current_image_path, current_depth = queue.pop(0)  # BFS
        if current_depth >= depth:
            continue
        if current_image_path in visited:
            continue
        visited.add(current_image_path)


        if model == 'resnet50':
            query_image = Image.open(query_image_path).convert('RGB')
            query_image_tensor = transform(query_image).unsqueeze(0).to(device)
            query_features_resnet = extract_features(query_image_tensor, resnet_model)
        
            similar_images = find_similar_images(query_features_resnet, features_resnet50,metric)



        if model == 'vgg16':
            query_image = Image.open(query_image_path).convert('RGB')
            query_image_tensor = transform(query_image).unsqueeze(0).to(device)
            query_features_vgg = extract_features(query_image_tensor, vgg_model)
        
            similar_images = find_similar_images(query_features_vgg, features_vgg16,metric)

        # Exclude the current image from similar_images
        similar_images = [sim_img for sim_img in similar_images if sim_img[0][0] != current_image_path]

        if not similar_images:
            continue  # Skip if no similar images found

        # Normalize similarities and add edges
        for similar_image in similar_images:
            similar_image_path = similar_image[0][0]
            # Add edge to the graph with normalized weight
            G.add_edge(current_image_path, similar_image_path, weight=similar_image[1])
            queue.append((similar_image_path, current_depth + 1))
    return G



from collections import defaultdict
import random
def stochastic_diffusion_process(G, query_image_path, num_steps=10, reset_prob=0.01, num_walks=1000):

    # Ensure the query image is in the graph
    if query_image_path not in G:
        raise ValueError("Query image path not found in the graph.")

    # Precompute cumulative distribution functions (CDFs) for neighbors
    neighbor_cdfs = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            weights = np.array([G[node][nbr].get('weight', 1.0) for nbr in neighbors])
            total_weight = weights.sum()
            probabilities = weights / total_weight
            cdf = np.cumsum(probabilities)
            neighbor_cdfs[node] = (neighbors, cdf)

    # Initialize a counter for visits to each node
    visit_counts = defaultdict(int)

    # Simulate random walks
    for walk in range(num_walks):
        current_node = query_image_path
        for step in range(num_steps):
            # Reset to query image with reset probability
            if random.random() < reset_prob:
                current_node = query_image_path

            # Record visit to current node
            visit_counts[current_node] += 1

            # Check if the current node has neighbors
            if current_node in neighbor_cdfs:
                neighbors, cdf = neighbor_cdfs[current_node]
                rnd = random.random()
                # Select the next node based on the CDF
                idx = np.searchsorted(cdf, rnd)
                current_node = neighbors[idx]
            else:
                # If no neighbors, stay in the current node
                continue

    # Calculate probabilities by normalizing visit counts
    total_visits = sum(visit_counts.values())
    node_probabilities = {node: count / total_visits for node, count in visit_counts.items()}

    return node_probabilities

# Usage example


    
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
    graph = 1
    # Load the query image
    query_image = Image.open(filepath).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze(0).to(device)

    if graph == 0:
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

    if graph == 1:
        G_resnet = build_similarity_graph(filepath, depth=3, N=5,metric=metric,model='resnet50')
        G_vgg = build_similarity_graph(filepath, depth=3, N=5,metric=metric,model='vgg16')
        num_steps = 5
        reset_prob = 0.01  # 1% probability to reset to the query image at each step
        num_walks = 100  # Number of random walks to simulate
        
        node_probabilities_resnet = stochastic_diffusion_process(
        G_resnet,
        filepath,
        num_steps=num_steps,
        reset_prob=reset_prob,
        num_walks=num_walks)
    
    
        sorted_nodes_resnet = sorted(node_probabilities_resnet.items(), key=lambda x: x[1], reverse=True)
        similar_images_resnet = sorted_nodes_resnet[:10]
    
        print(similar_images_resnet)
    
        node_probabilities_vgg = stochastic_diffusion_process(
        G_vgg,
        filepath,
        num_steps=num_steps,
        reset_prob=reset_prob,
        num_walks=num_walks)
    
    
        sorted_nodes_vgg = sorted(node_probabilities_vgg.items(), key=lambda x: x[1], reverse=True)
        similar_images_vgg = sorted_nodes_vgg[:10]

        print(similar_images_vgg)

        
    
    # Convert images to base64 and include their filenames
    images_resnet = []
    for img_array, similarity in similar_images_resnet:
        if graph == 0:
            img = Image.open(img_array[0])
        if graph == 1:
            img = Image.open(img_array)
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
        if graph == 0:
            img = Image.open(img_array[0])
        if graph == 1:
            img = Image.open(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_vgg.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array[0])  # Include file name
        })

    images_resnet_contrastive = []
    # for img_array, similarity in similar_images_resnet_contrastive:
    #     img = Image.open(img_array[0])
    #     buffered = io.BytesIO()
    #     img.save(buffered, format="PNG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode()
    #     images_resnet_contrastive.append({
    #         'img_data': img_str,
    #         'similarity': f"{similarity:.4f}",
    #         'name': os.path.basename(img_array[0])  # Include file name
    #     })

    images_vgg_contrastive = []
    # for img_array, similarity in similar_images_vgg_contrastive:
    #     img = Image.open(img_array[0])
    #     buffered = io.BytesIO()
    #     img.save(buffered, format="PNG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode()
    #     images_vgg_contrastive.append({
    #         'img_data': img_str,
    #         'similarity': f"{similarity:.4f}",
    #         'name': os.path.basename(img_array[0])  # Include file name
    #     })

    

    # Convert the uploaded image to base64
    with open(filepath, "rb") as f:
        uploaded_image_data = base64.b64encode(f.read()).decode()

    return render_template('results.html', images_resnet=images_resnet, images_vgg=images_vgg,images_resnet_contrastive=images_resnet_contrastive,images_vgg_contrastive=images_vgg_contrastive,
                           uploaded_image=uploaded_image_data, metric=metric, filename=filename)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)