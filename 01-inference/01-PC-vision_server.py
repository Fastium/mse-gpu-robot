import sys
import time
import base64
import zmq
import cv2
import torch
import torchvision.transforms as transforms
import argparse
from torchvision.models import mobilenet_v2
import torch.nn as nn

# --- CONFIGURATION ---
ZMQ_PORT = 5555
# INPUT: Use the ORIGINAL weights, not the TRT engine from Jetson
MODEL_PATH = "../models/w11-mobilenet_v2_b16_lr0.001_e40.pth.tar" 
NUM_CLASSES = 2

# Emulate Jetson Camera Specs
CAM_WIDTH = 320
CAM_HEIGHT = 224
MODEL_INPUT_SIZE = 224

# Select Device (PC might not have CUDA, use CPU if needed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Init] Running simulation on: {DEVICE}")

def get_model():
    print(f"[Model] Loading Standard PyTorch Model from {MODEL_PATH}...")
    
    # Re-instantiate the architecture (Must match training)
    model = mobilenet_v2()
    # Replace classifier head for 2 classes
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)

    # Load weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"[Error] Could not load weights: {e}")
        print("Make sure you are using the original .pth file, not the _trt.pth one!")
        sys.exit(1)

    model = model.to(DEVICE).eval()
    print(f"[Model] Model loaded.")
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='PC Vision Server Simulation')
    parser.add_argument('--input', type=str, default='0', help='Camera ID (0) or Video File path (demo.mp4)')
    parser.add_argument('--mirror', action='store_true', help='Activate mirror mode for webcam')
    args = parser.parse_args()

    # 1. Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Comms] ZMQ Publisher bound to port {ZMQ_PORT}")

    # 2. Setup Input Source (Webcam or Video)
    source = args.input
    # Check if source is a digit (Webcam ID)
    if source.isdigit():
        source = int(source)
        print(f"[Camera] Opening Webcam ID {source}...")
    else:
        print(f"[Video] Opening Video File {source}...")

    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("[Error] Could not open video source.")
        return

    # 3. Load Model
    model = get_model()
    preprocess = get_transform()

    print("[System] Starting Loop...")

    last_time = time.time()

    # Crop offsets (Same as Jetson)
    crops_x = {
        "left": 0,
        "center": 48,
        "right": 96
    }

    try:
        while True:
            ret, frame = cap.read()
            
            # --- VIDEO LOOPING LOGIC ---
            if not ret:
                if isinstance(source, str): 
                    # End of video file, loop back
                    print("[Video] Loop...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Error] Camera disconnected.")
                    break

            # --- RESIZE TO JETSON RESOLUTION ---
            # PC Webcams are usually 640x480 or 720p. 
            # We must resize to 320x224 to match the Jetson's perspective exactly.
            image = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

            # Flip for mirror effect
            if args.mirror:
                image = cv2.flip(image, 1)

            # --- FPS CALC ---
            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time
            fps = 1.0 / dt if dt > 0 else 0

            # --- PREP CROPS ---
            # Standard PyTorch usually expects BGR -> RGB conversion if trained with PIL
            # OpenCV gives BGR.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare batch
            tensors = []
            
            # Extract crops
            img_left = image_rgb[:, crops_x["left"]:crops_x["left"]+MODEL_INPUT_SIZE]
            img_center = image_rgb[:, crops_x["center"]:crops_x["center"]+MODEL_INPUT_SIZE]
            img_right = image_rgb[:, crops_x["right"]:crops_x["right"]+MODEL_INPUT_SIZE]

            # Convert to Tensors
            for img_crop in [img_left, img_center, img_right]:
                pil_img = transforms.ToPILImage()(img_crop)
                tensors.append(preprocess(pil_img).unsqueeze(0).to(DEVICE))

            # --- INFERENCE ---
            probs_map = {}
            keys = ["left", "center", "right"]

            with torch.no_grad(): # Essential for inference performance
                for i, input_tensor in enumerate(tensors):
                    output = model(input_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0][0] 
                    probs_map[keys[i]] = float(prob)

            # --- PUBLISH ---
            # Encode for Viewer
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "probs": probs_map,
                "prob_target": float(probs_map["center"]),
                "image_b64": jpg_as_text,
                "jetson_fps": float(fps)
            }
            socket.send_json(payload)

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()