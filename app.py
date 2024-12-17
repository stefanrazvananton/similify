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
import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import networkx as nx
from utils import FeatureExtractorResNet,FeatureExtractorVGG,FeatureExtractorResNetContrastive,FeatureExtractorVggContrastive
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# fiecare folder din "imgs" contine imagini dintr-o clasa
dataset = datasets.ImageFolder(root='imgs', transform=transform)



# incarc toate cele 4 modele si features 
resnet_model = FeatureExtractorResNet()
resnet_model = resnet_model.to(device)
resnet_model.eval()
features_resnet50 = np.load('features_resnet50.npy')

vgg_model = FeatureExtractorVGG()
vgg_model = vgg_model.to(device)
vgg_model.eval()
features_vgg16 = np.load('features_vgg16.npy')

resnet_contrastive_model = FeatureExtractorResNetContrastive()
resnet_contrastive_model = resnet_contrastive_model.to(device)
resnet_contrastive_model.eval()
resnet_contrastive_model.load_state_dict(torch.load('siamese_resnet50_9.pth', map_location=device))
features_resnet_contrastive = np.load('features_resnet_contrastive_9.npy')

vgg_contrastive_model = FeatureExtractorVggContrastive()
vgg_contrastive_model = vgg_contrastive_model.to(device)
vgg_contrastive_model.eval()
vgg_contrastive_model.load_state_dict(torch.load('siamese_vgg16_9.pth', map_location=device))
features_vgg_contrastive = np.load('features_vgg_contrastive_9.npy')

# metoda 1 de gasire, cei mai apropiati N vecini
def find_similar_images(query_features, features_db, metric):
    if metric == 'cosine':

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(features_db)
        distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)
        
        distances = distances[0]
        indices = indices[0] 

        similar_images = [(dataset.imgs[idx], 1 - distances[i]) for i, idx in enumerate(indices)]
    else:
        if metric == 'euclidean':
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean').fit(features_db)
            distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)
        elif metric == 'manhattan':
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='manhattan').fit(features_db)
            distances, indices = nbrs.kneighbors(query_features.reshape(1, -1), n_neighbors=10)

        distances = distances[0]
        indices = indices[0] 
        
        similar_images = [(dataset.imgs[idx], distances[i]) for i, idx in enumerate(indices)]

    return similar_images

# metoda 2 de gasire, plecand de la imaginea de query ma plimb pe un graf
# inspirat de aici https://github.com/fyang93/diffusion/tree/master
def build_similarity_graph(query_image_path, depth, N,metric,model):
    G = nx.Graph()
    visited = set()
    queue = [(query_image_path, 0)]
    while queue:
        current_image_path, current_depth = queue.pop(0)
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


        if model == 'resnet50_contrastive':
            query_image = Image.open(query_image_path).convert('RGB')
            query_image_tensor = transform(query_image).unsqueeze(0).to(device)
            query_features_resnet_contrastive = extract_features(query_image_tensor, resnet_contrastive_model)
        
            similar_images = find_similar_images(query_features_resnet_contrastive, features_resnet_contrastive,metric)


        if model == 'vgg16_contrastive':
            query_image = Image.open(query_image_path).convert('RGB')
            query_image_tensor = transform(query_image).unsqueeze(0).to(device)
            query_features_vgg_contrastive = extract_features(query_image_tensor, vgg_contrastive_model)
        
            similar_images = find_similar_images(query_features_vgg_contrastive, features_vgg_contrastive,metric)
        
    
        similar_images = [sim_img for sim_img in similar_images if sim_img[0][0] != current_image_path]

        if not similar_images:
            continue


        for similar_image in similar_images:
            similar_image_path = similar_image[0][0]
            G.add_edge(current_image_path, similar_image_path, weight=similar_image[1])
            queue.append((similar_image_path, current_depth + 1))
    return G


# procesul de random walk prin care decid care imagini sunt relevante pentru imaginea de query
from collections import defaultdict
import random
def stochastic_diffusion_process(G, query_image_path, num_steps=10, reset_prob=0.01, num_walks=1000):


    if query_image_path not in G:
        raise ValueError("Query image path not found in the graph.")


    neighbor_cdfs = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            weights = np.array([G[node][nbr].get('weight', 1.0) for nbr in neighbors])
            total_weight = weights.sum()
            probabilities = weights / total_weight
            cdf = np.cumsum(probabilities)
            neighbor_cdfs[node] = (neighbors, cdf)


    visit_counts = defaultdict(int)


    for walk in range(num_walks):
        current_node = query_image_path
        for step in range(num_steps):

            if random.random() < reset_prob:
                current_node = query_image_path


            visit_counts[current_node] += 1

            if current_node in neighbor_cdfs:
                neighbors, cdf = neighbor_cdfs[current_node]
                rnd = random.random()
                idx = np.searchsorted(cdf, rnd)
                current_node = neighbors[idx]
            else:
                continue


    total_visits = sum(visit_counts.values())
    node_probabilities = {node: count / total_visits for node, count in visit_counts.items()}

    return node_probabilities

def extract_features(image, model):
    with torch.no_grad():
        features = model(image)
        features = features.cpu().numpy().flatten()

        return features





@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files or 'metric' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        metric = request.form['metric']
        graph = request.form['graph_search']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('display_results', filename=file.filename, metric=metric, graph=graph))
    return render_template('upload.html')



@app.route('/results/<filename>')
def display_results(filename):
    metric = request.args.get('metric', 'cosine')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    graph = request.args.get('graph','True')

    query_image = Image.open(filepath).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze(0).to(device)

    if graph == 'false':

        query_features_resnet = extract_features(query_image_tensor, resnet_model)
        query_features_vgg = extract_features(query_image_tensor, vgg_model)
        query_features_resnet_contrastive = extract_features(query_image_tensor, resnet_contrastive_model)
        query_features_vgg_contrastive = extract_features(query_image_tensor, vgg_contrastive_model)
    
        similar_images_resnet = find_similar_images(query_features_resnet, features_resnet50, metric)
        similar_images_vgg = find_similar_images(query_features_vgg, features_vgg16, metric)
        similar_images_resnet_contrastive = find_similar_images(query_features_resnet_contrastive, features_resnet_contrastive, metric)
        similar_images_vgg_contrastive = find_similar_images(query_features_vgg_contrastive, features_vgg_contrastive, metric)

    if graph == 'true':
        G_resnet = build_similarity_graph(filepath, depth=3, N=5,metric=metric,model='resnet50')
        G_vgg = build_similarity_graph(filepath, depth=3, N=5,metric=metric,model='vgg16')
        G_resnet_contrastive = build_similarity_graph(filepath, depth=3, N=5,metric=metric,model='resnet50_contrastive')
        G_vgg_contrastive = build_similarity_graph(filepath, depth=3, N=5,metric=metric,model='vgg16_contrastive')        

        
        num_steps = 5
        reset_prob = 0.01 
        num_walks = 100
        
        node_probabilities_resnet = stochastic_diffusion_process(
        G_resnet,
        filepath,
        num_steps=num_steps,
        reset_prob=reset_prob,
        num_walks=num_walks)
    
    
        sorted_nodes_resnet = sorted(node_probabilities_resnet.items(), key=lambda x: x[1], reverse=True)
        similar_images_resnet = sorted_nodes_resnet[:10]
    
        node_probabilities_vgg = stochastic_diffusion_process(
        G_vgg,
        filepath,
        num_steps=num_steps,
        reset_prob=reset_prob,
        num_walks=num_walks)
    
    
        sorted_nodes_vgg = sorted(node_probabilities_vgg.items(), key=lambda x: x[1], reverse=True)
        similar_images_vgg = sorted_nodes_vgg[:10]

        node_probabilities_resnet_contrastive = stochastic_diffusion_process(
        G_resnet_contrastive,
        filepath,
        num_steps=num_steps,
        reset_prob=reset_prob,
        num_walks=num_walks)
    
    
        sorted_nodes_resnet_contrastive = sorted(node_probabilities_resnet_contrastive.items(), key=lambda x: x[1], reverse=True)
        similar_images_resnet_contrastive = sorted_nodes_resnet_contrastive[:10]
        #print(similar_images_resnet_contrastive)

        node_probabilities_vgg_contrastive = stochastic_diffusion_process(
        G_vgg_contrastive,
        filepath,
        num_steps=num_steps,
        reset_prob=reset_prob,
        num_walks=num_walks)
    
    
        sorted_nodes_vgg_contrastive = sorted(node_probabilities_vgg_contrastive.items(), key=lambda x: x[1], reverse=True)
        similar_images_vgg_contrastive = sorted_nodes_vgg_contrastive[:10]
        #print(similar_images_vgg_contrastive)

    

    images_resnet = []
    for img_array, similarity in similar_images_resnet:
        if graph == 'false':
            img_array = img_array[0]
        if graph == 'true':
            img_array = img_array
        img = Image.open(img_array)    
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_resnet.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array)
        })

    images_vgg = []
    for img_array, similarity in similar_images_vgg:
        if graph == 'false':
            img_array = img_array[0]
        if graph == 'true':
            img_array = img_array
        img = Image.open(img_array) 
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_vgg.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array)
        })

    images_resnet_contrastive = []
    for img_array, similarity in similar_images_resnet_contrastive:
        if graph == 'false':
            img_array = img_array[0]
        if graph == 'true':
            img_array = img_array
        img = Image.open(img_array) 
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_resnet_contrastive.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array)
        })

    images_vgg_contrastive = []
    for img_array, similarity in similar_images_vgg_contrastive:
        if graph == 'false':
            img_array = img_array[0]
        if graph == 'true':
            img_array = img_array
        img = Image.open(img_array) 
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images_vgg_contrastive.append({
            'img_data': img_str,
            'similarity': f"{similarity:.4f}",
            'name': os.path.basename(img_array)
        })

    with open(filepath, "rb") as f:
        uploaded_image_data = base64.b64encode(f.read()).decode()

    return render_template('results.html', images_resnet=images_resnet, images_vgg=images_vgg, images_resnet_contrastive=images_resnet_contrastive, images_vgg_contrastive=images_vgg_contrastive,
                           uploaded_image=uploaded_image_data, metric=metric, filename=filename)



if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)