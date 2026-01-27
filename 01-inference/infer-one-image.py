import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from reshape import reshape_model

def default_transform(resolution):
    return transforms.Compose([
        transforms.Resize(resolution),  #change resolution sans deformation
		transforms.CenterCrop(resolution),#crop le côter resté trop grand
        transforms.ToTensor(),          # Convert the image to a float-tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the color
    ])

def infer_one_image(model_path: str, image_path: str) -> torch.Tensor:
    """Infers over a single image with the specified model.

    Args:
        model_path (str): The path to the model i.e. "models/googlenet/model_best.pth.tar".
        image_path (str): The path to the image i.e. "data/test/not_target.jpg".

    Returns:
        A Tensor (CPU) of proba, The size is the number of classes.
    """

    # Load the checkpoint from extracted tar file
    checkpoint = torch.load(model_path)

    # Image normalization
    resolution = checkpoint["resolution"]
    transform = default_transform(resolution)

    # Create Model from architecture
    modelname = checkpoint["arch"]
    model = models.__dict__[modelname]()

    # Reshape model to the number of classes
    num_classes = 2
    model = reshape_model(model, checkpoint["arch"], num_classes, logger=False)

    # Apply the trained weights (checkpoint to model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)  # strict error if incompatible

    # Transfer model to GPU
    device = torch.device(0)  # GPU 0
    model = model.cuda(device)

    # switch to test mode
    model.eval()

    # Infer over the dataset
    with torch.no_grad():
        with Image.open(image_path) as image:
            # Transform image
            tensor = transform(image)
            tensor = tensor.unsqueeze(0)
            # Send images to the GPU
            tensorGM = tensor.to(device)

            # Predicts
            output = model(tensorGM)

            # Convert to probability
            probas = F.softmax(output, dim=1)

            return probas.cpu()

if __name__ == "__main__":
    probas = infer_one_image("alexnet.pth.tar", "25.jpg")
    print(f"[PYTORCH] : Alexnet probas (cat,dog) : {probas}")
    assert abs(1 - torch.sum(probas)) < 0.0001

    indexWinner = probas.argmax().item()
    print("[PYTORCH] : Alexnet predicted:", "cat" if indexWinner == 0 else "dog")
