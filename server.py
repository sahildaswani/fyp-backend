import sys
sys.path.append('/Users/sahildaswani/opt/anaconda3/envs/fyp-web/lib/python3.11/site-packages')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from timm.models.vision_transformer import vit_small_patch16_224
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
cors = CORS(app)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load the PyTorch models
has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")

classes = ['Cataract', 'Myopia', 'Normal']

# Load the ResNet50 model
def load_resnet50_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)
    model.load_state_dict(torch.load('resnet50.pth', map_location=device))
    model.eval()

    return model

resnet50_model = load_resnet50_model()

# Load the VGG16 model
def load_vgg16_model():
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, len(classes))
    model = model.to(device)
    model.load_state_dict(torch.load('vgg16.pth', map_location=device))
    model.eval()

    return model

vgg16_model = load_vgg16_model()

# Load the ViT model
def load_vit_model():
    model = vit_small_patch16_224(pretrained=True)

    # Modify the head of the model for binary classification
    model.head = nn.Linear(model.head.in_features, len(classes))
    model = model.to(device)
    model.load_state_dict(torch.load('vit.pth', map_location=device))
    model.eval()

    return model

vit_model = load_vit_model()

# Define the image processing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def cropImage(img, crop):
    # Crop the image
    width, height = img.size
    left = int(crop['x'] * width)
    top = int(crop['y'] * height)
    right = int((crop['x'] + crop['width']) * width)
    bottom = int((crop['y'] + crop['height']) * height)
    img = img.crop((left, top, right, bottom))
    return img

# Define the prediction function for each model
def predict(model):
    # Get the uploaded image
    image = request.files['image']
    crop = json.loads(request.form['crop'])
    print(crop['x'], crop['y'], crop['width'], crop['height'])
    
    # # Open the image using PIL and apply the processing pipeline
    img = Image.open(image)
    img = cropImage(img, crop)

    img_tensor = preprocess(img).float()
    img_tensor.unsqueeze_(0) # Add a batch dimension

    if torch.backends.mps.is_available():
            img_tensor = img_tensor.to(device)
    
    input=Variable(img_tensor)
    output=model(input).cpu()
    index=output.data.numpy().argmax()
    pred=classes[index]
    
    # Return the predicted class as a response
    return pred;

@app.route('/predict/resnet50', methods=['POST'])
def predict_resnet50():
    return jsonify({'class': predict(resnet50_model), 'model': 'resnet50'})

@app.route('/predict/vgg16', methods=['POST'])
def predict_vgg16():
    return jsonify({'class': predict(vgg16_model), 'model': 'vgg16'})

@app.route('/predict/vit', methods=['POST'])
def predict_vit():
    return jsonify({'class': predict(vit_model), 'model': 'vit'})

if __name__ == '__main__':
    app.run()

