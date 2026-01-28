# convert_to_trt.py
import torch
from torch2trt import torch2trt
from torchvision.models import resnet18
import torch.nn as nn
import time

# --- CONFIG ---
INPUT_MODEL = "../models/resnet18.pth.tar"
OUTPUT_TRT = "../models/resnet18_trt.pth"
NUM_CLASSES = 2
DEVICE = torch.device('cuda')

print(f"[Init] Loading PyTorch model from {INPUT_MODEL}...")

# 1. Load Original Model
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load weights (CPU first to be safe)
checkpoint = torch.load(INPUT_MODEL, map_location='cpu')
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE).eval()

# 2. Create Dummy Input
# TensorRT needs to know the exact input shape to optimize memory
x = torch.ones((1, 3, 224, 224)).to(DEVICE)

print("[Convert] Starting TensorRT conversion (FP16)...")
print("This can take 1-3 minutes on Jetson Nano. Be patient.")

start = time.time()

# --- MAGIC HAPPENS HERE ---
# fp16_mode=True doubles performance on Jetson Nano (Maxwell GPU)
model_trt = torch2trt(model, [x], fp16_mode=True)

end = time.time()
print(f"[Success] Conversion done in {end - start:.2f} seconds.")

# 3. Save the Optimized Engine
print(f"[Save] Saving TRT model to {OUTPUT_TRT}...")
torch.save(model_trt.state_dict(), OUTPUT_TRT)

print("Done. You can now use this optimized model.")
