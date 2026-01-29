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
MODEL_PATH = "../models/mobilenet_v2_b16_lr0.001_e40_trt.pth"
DEVICE = torch.device("cuda")

CAM_WIDTH = 320
CAM_HEIGHT = 224
MODEL_INPUT_SIZE = 224

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
        camera = CSICamera(width=CAM_WIDTH, height=CAM_HEIGHT, capture_width=1280, capture_height=720, capture_fps=30)
        camera.running = True
        print("[Camera] Ready.")
    except Exception as e:
        print(f"[Error] Camera init failed: {e}")
        return

    # 3. Load Model
    model = get_model()
    preprocess = get_transform()

    print("[System] Starting Multi-Crop Loop...")

    last_time = time.time()

    crops_x = {
        "left": 0,
        "center": 48,
        "right": 96
    }

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

            # --- PREP CROPS ---
            # Prepare the inputs
            batch_tensors = []

            # Note: image is in HWC format (Height, Width, Channel)
            # Cut out the 3 zones
            img_left = image[:, crops_x["left"]:crops_x["left"]+MODEL_INPUT_SIZE]
            img_center = image[:, crops_x["center"]:crops_x["center"]+MODEL_INPUT_SIZE]
            img_right = image[:, crops_x["right"]:crops_x["right"]+MODEL_INPUT_SIZE]

            # Transform into tensors
            tensors = []
            for img_crop in [img_left, img_center, img_right]:
                pil_img = transforms.ToPILImage()(img_crop)
                tensors.append(preprocess(pil_img).unsqueeze(0).to(DEVICE))

            # --- INFERENCE ---
            # To avoid breaking the TRT engine (often compiled with batch_size=1),
            # we do 3 sequential passes. This is safe and fast on Nano.
            probs_map = {}
            keys = ["left", "center", "right"]

            for i, input_tensor in enumerate(tensors):
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0][0] # Index 0 = Target? Check your index!
                # WARNING: If your model outputs [NoTarget, Target], then target prob is index 1.
                # I assume index 0 here based on your previous script, but verify this.
                probs_map[keys[i]] = float(prob)

            # --- PUBLISH ---
            # Send the complete LARGE image for the viewer
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "probs": probs_map, # Ex: {"left": 0.1, "center": 0.9, "right": 0.2}
                "prob_target": float(probs_map["center"]),
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
