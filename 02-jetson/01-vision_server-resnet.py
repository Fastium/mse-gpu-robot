import sys
import time
import base64
import gc
import zmq
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from jetcam.csi_camera import CSICamera

# --- CONFIGURATION ---
ZMQ_PORT = 5555
MODEL_PATH = "../models/resnet18.pth.tar"
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    print(f"[Model] Loading {MODEL_PATH} on {DEVICE}...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE).eval()

    # Cleanup and Warmup
    gc.collect()
    torch.cuda.empty_cache()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    model(dummy)
    print("[Model] Ready.")
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    # 1. ZMQ Setup
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # We use a High Water Mark (HWM) of 2 to drop old frames if network lags
    # This ensures real-time latency over throughput
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Comms] ZMQ Publisher bound to port {ZMQ_PORT}")

    # 2. Camera Setup
    print("[Camera] Starting stream...")
    camera = CSICamera(width=224, height=224, capture_width=1080, capture_height=720, capture_fps=30)
    camera.running = True

    # 3. Model Setup
    model = get_model()
    preprocess = get_transform()

    print("[System] Inference Loop Started.")

    last_time = time.time()

    try:
        while True:
            image = camera.value
            if image is None:
                continue

            # --- FPS CALCULATION (START) ---
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            # Avoid division by zero on first frame
            fps = 1.0 / dt if dt > 0 else 0.0

            # --- INFERENCE ---
            image_pil = transforms.ToPILImage()(image)
            input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            # --- PACKAGING ---
            # Compress image to JPEG
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "prob_target": float(probs[0]),
                "image_b64": jpg_as_text,
                "jetson_fps": float(fps) # Sending the real computed FPS
            }
            socket.send_json(payload)

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    finally:
        camera.running = False
        camera.cap.release()

if __name__ == "__main__":
    main()
