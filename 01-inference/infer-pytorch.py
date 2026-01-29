import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# =======================
# CONFIGURATION
# =======================
MODEL_NAME = "mobilenet_v2"
IMAGE_PATH = "../data/cible/Image_2025_0005_10_cible.jpg"
NUM_CLASSES = 2
# =======================
CHECKPOINT_PATH = f"../models/{MODEL_NAME}/model_best.pth.tar"
# =======================

def default_transform(resolution=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resolution),  # change resolution sans deformation
        transforms.CenterCrop(resolution),  # crop le côter resté trop grand
        transforms.ToTensor(),  # Convert the image to a float-tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the colors
    ])

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = models.__dict__[MODEL_NAME](weights=None)

# Modify the last layer to match number of classes
if hasattr(model, 'fc'):
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
elif hasattr(model, 'classifier'):
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)

# Load trained checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Handle checkpoint format (either state_dict directly or wrapped in 'state_dict' key)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

# Move model to device and set to evaluation mode
model = model.to(device)
model.eval()

# Load image
image = Image.open(IMAGE_PATH)

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

#check somme proba quasi 1
assert abs(1 - sum(probas)) < 0.0001

# Print probabilities with labels
print("-" * 30)
print(f"Target:     {probas[0]*100:.2f}%")
print(f"Not Target: {probas[1]*100:.2f}%")
print("-" * 30)
