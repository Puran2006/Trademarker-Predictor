import os
import torch
import time
import pandas as pd
import logging
from flask import Flask, request, jsonify
from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to track user requests
user_requests = defaultdict(int)

# Maximum allowed requests per user
MAX_REQUESTS = 5

# Load the model
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def load_model(model_path):
    # Define your model architecture
    input_dim = 5000  # Change this to your feature dimension
    hidden_dim1 = 128
    hidden_dim2 = 64
    output_dim = 45  # Number of classes

    model = SimpleNN(input_dim, hidden_dim1, hidden_dim2, output_dim)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    return model

# Load your vectorizer and model
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
train_data_path = 'idmanual.csv'
train_data = pd.read_csv(train_data_path)
vectorizer.fit(train_data['Description'])
model_path = 'model_torch_mlflow.pth'
model = load_model(model_path)

# Trademark classes mapping
trademark_classes = {
    1: "Chemicals",
    2: "Paints",
    3: "Cleaning Substances",
    4: "Industrial Oils",
    5: "Pharmaceuticals",
    6: "Common Metals",
    7: "Machines",
    8: "Hand Tools",
    9: "Computers & Scientific Devices",
    10: "Medical Supplies",
    11: "Appliances",
    12: "Vehicles",
    13: "Firearms",
    14: "Precious Metals",
    15: "Musical Instruments",
    16: "Paper Goods",
    17: "Rubber Products",
    18: "Leather Goods",
    19: "Building Materials",
    20: "Furniture",
    21: "Ropes and Textile Products",
    22: "Household Utensils",
    23: "Yarns and Threads",
    24: "Textiles",
    25: "Clothing",
    26: "Lace and Embroidery",
    27: "Carpets",
    28: "Games and Sporting Goods",
    29: "Meat, Fish, Poultry",
    30: "Coffee, Flour, Rice",
    31: "Grains, Agriculture",
    32: "Beers and Beverages",
    33: "Alcoholic Beverages",
    34: "Tobacco Products",
    35: "Advertising & Business Services",
    36: "Insurance & Finance Services",
    37: "Construction & Repair Services",
    38: "Telecommunications Services",
    39: "Shipping & Travel Services",
    40: "Material Treatment Services",
    41: "Education & Entertainment Services",
    42: "Science & Technology Services",
    43: "Food Services",
    44: "Medical & Vet Services",
    45: "Legal & Security Services"
}

# Define a function to preprocess text and make predictions
def predict(texts):
    # Transform the texts using the vectorizer
    X = vectorizer.transform(texts).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map predictions to trademark classes
    predicted_classes = [trademark_classes[pred] for pred in predicted.numpy().tolist()]

    return predicted_classes

@app.route('/predict', methods=['POST'])
def predict_route():
    user_id = request.json.get('user_id', None)
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    # Check and update request count
    if user_requests[user_id] >= MAX_REQUESTS:
        logging.warning(f"User {user_id} has exceeded the maximum number of requests.")
        return jsonify({'error': 'Too many requests'}), 429
    
    user_requests[user_id] += 1

    # Record inference time
    start_time = time.time()

    data = request.json
    texts = data.get('texts', [])
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    # Make predictions
    predictions = predict(texts)

    inference_time = time.time() - start_time
    logging.info(f"User {user_id} made a prediction request. Inference time: {inference_time:.4f} seconds")

    return jsonify({'predictions': predictions, 'inference_time': inference_time})


@app.route('/')
def home():
    return "TradeMarker Classes!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
