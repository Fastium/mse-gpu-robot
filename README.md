# MSE GPU Robot

This repository contains code and resources for the project of GPU.

## Usage

Follow the next steps to set up the project:
1. Create a virtual environment:
```bash
uv venv --python 3.12
source .venv/bin/activate
```

2. Install the required packages:
```bash
uv pip install -r requirements.txt
```
## Training
Copy the data (robot.zip) in the data folder and decompress it. Then run our script:
```bash
python splitTrainTestVal.py
```
It creates the folders `train/`, `test/` and `val/` with the corresponding images forf the training script.

To train different models there is a bash script `train-model.sh` that produce all necessary files to get models in `.onnx` files. All models are saved in the `models/` folder.
```bash
./00-training/train-model.sh
```

We can vizualize the training process with TensorBoard:
```bash
tensorboard --logdir=./models
```

## Local inference


## Jetson inference

The system utilizes a **Micro-services architecture** powered by **ZeroMQ (ZMQ)**. Instead of a single monolithic script, the workload is distributed across three specialized processes using a Publisher-Subscriber pattern.

1.  **Publisher (Jetson)**: The Vision Server. It captures images and performs continuous inference.
2.  **Subscriber A (Jetson)**: The Controller. It listens for detection probabilities and manages the motors.
3.  **Subscriber B (Laptop)**: The Viewer. It listens for video data, renders the overlay, and records to disk.

### Script Descriptions

#### `01-vision_server.py` (Runs on: **Jetson**)
* **Role:** The **Continuous Inference Engine**.
* **Details:** This script is the heavy lifter of the project. Because initializing the CSI camera and loading the PyTorch model into the GPU memory takes significant time and resources, this script is designed to run once and stay alive.
* **Workflow:**
    * It initializes the hardware and loads the neural network weights.
    * It enters a high-performance infinite loop.
    * In every iteration, it captures a frame, preprocesses it, runs the inference, and calculates the **real-time internal FPS**.
    * It broadcasts the results (Target Probability, System FPS, and the compressed image) to the local network via port `5555`.

#### `02-PC-web_viewer.py` (Runs on: **Laptop**)
* **Role:** **Telemetry & Recording**.
* **Details:** Running visualization on the Jetson consumes CPU cycles needed for AI. This script offloads that work to your laptop.
* **Workflow:**
    * It connects to the Jetson over Wi-Fi.
    * It decodes the incoming JPEG stream.
    * It draws the HUD overlay (Target Confidence + Inference FPS) and records the session to an `.avi` file on your laptop's disk.

#### `03-control.py` (Runs on: **Jetson**)
* **Role:** The **Logic Controller**.
* **Details:** This is a lightweight script containing the robot's behavior.
* **Workflow:**
    * It connects to the `vision_server`'s data stream.
    * It receives processed data with near-zero latency.
    * It executes logic (e.g., `if probability > 0.75: stop`).
* **Key Benefit:** Since this script does not hold the model in memory, you can stop, edit, and restart it instantly to tweak parameters (speed, thresholds) without restarting the entire vision pipeline.

### Setup & Usage

#### Step 1: Start the Vision Engine (Jetson)
This process initializes the AI. It takes a moment to warm up.
```bash
# On the Jetson
python3 01-vision_server.py
```
Wait for the message: [System] Inference Loop Started.

#### Step 2 (optionnal): Start the Visualization (Laptop)

Verify the video feed and telemetry before enabling motors.

1. Open web_viewer_pc_final.py and set JETSON_IP = "192.168.1.XX".
2. Run the script:
  ```Bash
  # On the Laptop
  python 02-PC-web_viewer.py
  ```
  Open your browser to http://localhost:5000.

#### Step 3: Activate Control (Jetson)

Once the vision is stable, launch the control logic in a separate terminal.
```bash
# On the Jetson (New Terminal)
python3 control.py
```
