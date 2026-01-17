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
It contains the source code in a jupyter notebook, but you must have the robot to run it because of the hardware specific.
