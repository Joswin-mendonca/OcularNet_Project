from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load model
model = torch.load("model.pth", map_location="cpu")  # or load state_dict if that's how you saved it
model.eval()

# Image transforms (should match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in templates/

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    img = Image.open(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return f'Predicted class: {predicted.item()}'

if __name__ == '__main__':
    app.run(debug=True)
