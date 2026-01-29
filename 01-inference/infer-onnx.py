import onnxruntime as ort
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# =======================
# CONFIGURATION
# =======================
MODEL_NAME = "mobilenet_v2"
IMAGE_PATH = "../data/cible/Image_2025_0005_10_cible.jpg"
NUM_CLASSES = 2
# =======================
ONNX_MODEL_PATH = f"../models/{MODEL_NAME}/{MODEL_NAME}.onnx"
# =======================

def default_transform(resolution=(224,224)):
    return transforms.Compose([
        transforms.Resize(resolution),  #change resolution sans deformation
    	transforms.CenterCrop(resolution),#crop le côter resté trop grand
        transforms.ToTensor(),          # Convert the image to a float-tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the colors
    ])

providers = ["CUDAExecutionProvider"]

# create ONNX session using CUDA
ort_sess = ort.InferenceSession(ONNX_MODEL_PATH, providers = providers)

# Load image
image = Image.open(IMAGE_PATH)
# adapter l’image pour la rendre compatible avec ce qu’attends l’entrée du modèle
transform = default_transform() #voir annexe
# transform image into tensor
input_tensor = transform(image)
# change shape of tensor and transfer it to the device
input_tensor = input_tensor.unsqueeze(0).to("cuda:0")


io_binding = ort_sess.io_binding()
# bind input_tensor to the model
io_binding.bind_input(
    name = "input_0",
    device_type = "cuda",
    device_id = 0,
    element_type = np.float32,
    shape = tuple(input_tensor.shape),
    buffer_ptr = input_tensor.data_ptr()
)

# create a tensor for the output on the device
output_tensor = torch.empty((1, NUM_CLASSES), dtype = torch.float32, device =
"cuda:0").contiguous()
# bind the output to the model
io_binding.bind_output(
    name = "output_0",
    device_type = "cuda",
    device_id = 0,
    element_type = np.float32,
    shape = tuple(output_tensor.shape),
    buffer_ptr = output_tensor.data_ptr()
)

ort_sess.run_with_iobinding(io_binding)
#recupération des probas cotés host
probas = output_tensor.to("cpu")[0].tolist()

#check somme proba quasi 1
assert abs(1 - sum(probas)) < 0.0001

# Print probabilities with labels
print("-" * 30)
print(f"Target:     {probas[0]*100:.2f}%")
print(f"Not Target: {probas[1]*100:.2f}%")
print("-" * 30)


