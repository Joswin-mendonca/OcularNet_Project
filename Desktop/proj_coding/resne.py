import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# Load a pretrained ResNet-18 model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()  # Set to evaluation mode

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),           # Resize to 256x256
    transforms.CenterCrop(224),       # Crop to 224x224 (for ResNet)
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load image and convert to RGB
image = Image.open("robot.webp").convert("RGB")

# Apply the transformations and save the RGB image
# Save the RGB-converted image before transformation
transformed_image = transform(image)  # Apply transformations
image.save("converted_image.jpg")  # Save the original image in RGB format

# Convert the transformed image back to a PIL image (for visual verification if needed)
transformed_pil_image = transforms.ToPILImage()(transformed_image)

# Save the transformed image if needed
transformed_pil_image.save("transformed_image.jpg")

# Prepare input tensor
input_tensor = transformed_image.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the top prediction
_, predicted_class = torch.max(probabilities, 0)
print(predicted_class.item())

# Modify the final layer for 10 classes
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
