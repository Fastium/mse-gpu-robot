import sys
import os

# --- DEBUG: Trace imports to find the crash ---
print("[Init] Python started. Importing standard libs...", flush=True)
import time
import base64
import gc
import zmq

print("[Init] Importing OpenCV...", flush=True)
import cv2

print("[Init] Importing PyTorch (This usually takes time)...", flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F

print("[Init] Importing Torchvision...", flush=True)
import torchvision.transforms as transforms
import torchvision.models as models

print("[Init] Importing JetCam...", flush=True)
from jetcam.csi_camera import CSICamera

print("[Init] Imports done. Configuring...", flush=True)

# --- CONFIGURATION ---
ZMQ_PORT = 5555
MODEL_PATH = "../models/resnet18.pth.tar" # Ensure this path is correct relative to execution
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Init] Device selected: {DEVICE}", flush=True)

def get_model():
    print(f"[Model] constructing architecture...", flush=True)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    print(f"[Model] Loading weights from {MODEL_PATH}...", flush=True)
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found at {MODEL_PATH}", flush=True)
        sys.exit(1)

    # Load to CPU first to save GPU memory fragmentation
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"[Model] Moving to GPU...", flush=True)
    model = model.to(DEVICE).eval()

    # Aggressive memory cleanup
    print(f"[Model] Cleaning up RAM...", flush=True)
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    # Warmup
    print(f"[Model] Warmup inference...", flush=True)
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        model(dummy)

    print("[Model] Ready.", flush=True)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    # 1. Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    try:
        socket.bind(f"tcp://*:{ZMQ_PORT}")
        print(f"[Comms] ZMQ Publisher bound to port {ZMQ_PORT}", flush=True)
    except zmq.ZMQError as e:
        print(f"[Error] ZMQ Bind failed: {e}. Is the port already used?", flush=True)
        return

    # 2. Setup Camera
    print("[Camera] Initializing CSI Camera...", flush=True)
    try:
        camera = CSICamera(width=224, height=224, capture_width=1080, capture_height=720, capture_fps=30)
        camera.running = True
        print("[Camera] Camera pipeline ready.", flush=True)
    except Exception as e:
        print(f"[Error] Camera init failed: {e}", flush=True)
        return

    # 3. Load Model (Last step because it's heaviest)
    try:
        model = get_model()
        preprocess = get_transform()
    except Exception as e:
        print(f"[Error] Model loading failed: {e}", flush=True)
        camera.running = False
        return

    print("[System] Starting Inference Loop. Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            image = camera.value
            if image is None:
                continue

            # --- INFERENCE ---
            # Preprocess
            image_pil = transforms.ToPILImage()(image)
            input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            # --- PUBLISH ---
            # Compress image for network efficiency (visualization only)
            # Reduce quality to 50 to save bandwidth/CPU
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send data: Topic "vision" (implied), payload JSON
            payload = {
                "prob_target": float(probs[0]), # Assuming class 0 is target
                "image_b64": jpg_as_text
            }
            socket.send_json(payload)

            # Necessary to yield execution on single-core setups or tight loops
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[System] Stopping...", flush=True)
    except Exception as e:
        print(f"\n[Error] Loop crashed: {e}", flush=True)
    finally:
        camera.running = False
        print("[System] Cleanup done.", flush=True)

if __name__ == "__main__":
    main()
