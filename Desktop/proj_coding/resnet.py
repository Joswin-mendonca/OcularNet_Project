# resnet.py
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

# Load pretrained ResNet model without final layer (for features)
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final classification layer
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor).squeeze().numpy()
    return features / np.linalg.norm(features)  # normalize

def load_dataset_features(dataset_dir):
    features_db = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                feature = extract_features(path)
                features_db.append((filename, feature))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return features_db

def match_image_to_dataset(uploaded_image_path, dataset_dir='dataset'):
    uploaded_feat = extract_features(uploaded_image_path)
    dataset_features = load_dataset_features(dataset_dir)

    best_match = None
    best_score = -1

    for name, feat in dataset_features:
        similarity = np.dot(uploaded_feat, feat)  # cosine similarity
        if similarity > best_score:
            best_score = similarity
            best_match = name

    return best_match, best_score
