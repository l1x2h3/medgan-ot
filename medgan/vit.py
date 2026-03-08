import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

class TumorDetectionApp:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path

    def predict_image(self, image_path):
        # Load the model
        model = torchvision.models.vit_b_16()
        num_classes = 4  # Replace with the actual number of classes
        num_features = model.heads[0].in_features
        model.heads = torch.nn.Linear(num_features, num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        # Define image transformations
        IMG_SIZE = 224  # Ensure this matches the size used during training
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load and preprocess the image
        try:
            img = Image.open(image_path)
            img = transform(img).unsqueeze(0).to(self.device)  # Add batch dimension
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

        # Perform inference
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()
