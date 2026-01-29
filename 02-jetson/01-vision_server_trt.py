import sys
import time
import base64
import zmq
import cv2
import torch
import torchvision.transforms as transforms
from torch2trt import TRTModule # Import TensorRT wrapper
from jetcam.csi_camera import CSICamera

# --- CONFIGURATION ---
ZMQ_PORT = 5555
# We load the TRT optimized model, not the original PyTorch one
# MODEL_PATH = "../models/resnet18_trt.pth"
MODEL_PATH = "../models/mobilenet_v2_trt.pth"
DEVICE = torch.device("cuda")

print(f"[Init] Device selected: {DEVICE}")

def get_model():
    print(f"[Model] Loading TensorRT Engine from {MODEL_PATH}...")

    # Initialize the specific TRT Module wrapper
    model_trt = TRTModule()

    # Load the state dictionary (the compiled engine)
    model_trt.load_state_dict(torch.load(MODEL_PATH))

    print(f"[Model] Engine loaded on GPU.")
    return model_trt

def get_transform():
    # Standard ResNet preprocessing
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    # 1. Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"[Comms] ZMQ Publisher bound to port {ZMQ_PORT}")

    # 2. Setup Camera
    print("[Camera] Initializing CSI Camera...")
    try:
        camera = CSICamera(width=224, height=224, capture_width=1080, capture_height=720, capture_fps=30)
        camera.running = True
        print("[Camera] Ready.")
    except Exception as e:
        print(f"[Error] Camera init failed: {e}")
        return

    # 3. Load Model
    model = get_model()
    preprocess = get_transform()

    print("[System] Starting High-Performance Loop...")

    last_time = time.time()

    try:
        while True:
            image = camera.value
            if image is None:
                continue

            # --- FPS CALC ---
            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time
            fps = 1.0 / dt if dt > 0 else 0

            # --- INFERENCE ---
            image_pil = transforms.ToPILImage()(image)
            input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)

            # TensorRT Execution (As simple as calling the original model)
            output = model(input_tensor)

            # Post-process (Softmax) - Output is a standard Torch tensor
            # Note: TRT output might not be named, simply get index 0
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

            # --- PUBLISH ---
            # Compress to JPEG
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "prob_target": float(probs[0]),
                "image_b64": jpg_as_text,
                "jetson_fps": float(fps)
            }
            socket.send_json(payload)

    except KeyboardInterrupt:
        print("\n[System] Stopping...")
    finally:
        camera.running = False
        camera.cap.release()

if __name__ == "__main__":
    main()
