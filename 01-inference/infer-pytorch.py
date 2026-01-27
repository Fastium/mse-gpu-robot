import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def default_transform(resolution=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resolution),  # change resolution sans deformation
        transforms.CenterCrop(resolution),  # crop le côter resté trop grand
        transforms.ToTensor(),  # Convert the image to a float-tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the colors
    ])

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model configuration
model_name = "resnet18"
num_classes = 2

# Load model architecture
model = models.__dict__[model_name](weights=None)

# Modify the last layer to match number of classes
if hasattr(model, 'fc'):
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
elif hasattr(model, 'classifier'):
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)

# Load trained checkpoint
checkpoint_path = "models/resnet18/model_best.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle checkpoint format (either state_dict directly or wrapped in 'state_dict' key)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

# Move model to device and set to evaluation mode
model = model.to(device)
model.eval()

# Load image
image = Image.open("data/cible/Image_2025_0005_10_cible .jpg")
# image = Image.open("data/nocible/Image_2025_0005_28_nocible.jpg")

# Prepare image for model
transform = default_transform()
input_tensor = transform(image)

# Add batch dimension and move to device
input_tensor = input_tensor.unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    # Apply softmax to get probabilities
    probas = torch.softmax(output, dim=1)[0].tolist()

# Print probabilities
print(probas)

# Check that probabilities sum to ~1
assert abs(1 - sum(probas)) < 0.0001, f"Probabilities don't sum to 1: {sum(probas)}"
print(f"Inference successful! Probabilities sum to: {sum(probas)}")
