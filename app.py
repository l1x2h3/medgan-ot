import os
import io
import torch
import base64
import zipfile
from flask import Flask, request, jsonify, render_template, send_file
from medgan.dcgan import Generator_DCGAN, generate_examples_DCGAN
from medgan.progan import Generator_ProGAN, generate_examples_ProGAN, seed_everything
from medgan.stylegan import Generator_SG2, MappingNetwork, generate_examples_SG2
from medgan.vit import TumorDetectionApp
from medgan.wgan import Generator_WGAN, generate_examples_WGAN

# Initialize Flask app
app = Flask(__name__)

# Set seeds for reproducibility
seed_everything()

# Constants
Z_DIM = 256
FEATURES_GEN = 64
CHANNELS_IMG = 3
progan_steps = 6  # Number of steps for ProGAN fade-in
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set MEDGAN_PREVIEW_MODE=1 to launch UI without loading model checkpoints.
PREVIEW_MODE = os.getenv("MEDGAN_PREVIEW_MODE", "0") == "1"

# Model paths
model_paths = {
    "DCGAN": {
        "Glioma": "models/DCGAN-Glioma.pth",
        "Pituitary": "models/DCGAN-Meningioma.pth",
        "Meningioma": "models/DCGAN-Pituitary.pth",
    },
    "ProGAN": {
        "Glioma": "models/ProGAN-Glioma.pth",
        "Meningioma": "models/ProGAN-Meningioma.pth",
        "Pituitary": "models/ProGAN-Pituitary.pth",
    },
    "StyleGAN2": {
        "Glioma": {
            "generator": "models/StyleGAN2-Glioma.pth",
            "mapping": "models/StyleGAN2-Glioma-MappingNet.pth"
        },
        "Meningioma": {
            "generator": "models/StyleGAN2-Meningioma.pth",
            "mapping": "models/StyleGAN2-Meningioma-MappingNet.pth"
        },
        "Pituitary": {
            "generator": "models/StyleGAN2-Pituitary.pth",
            "mapping": "models/StyleGAN2-Pituitary-MappingNet.pth"
        },
    },
    "WGANs": {
        "Glioma": "models/WGAN-Glioma.pth",
        "Meningioma": "models/WGAN-Meningioma.pth",
        "Pituitary": "models/WGAN-Pituitary.pth",
    }
}


# Load DCGAN models
dcgan_generators = {}
for label, path in model_paths["DCGAN"].items():
    model = Generator_DCGAN(1, 256, 64, 3).to(torch.device('cpu'))  # Corrected Z_DIM to 256
    # Preview mode skips weight loading so the web UI can start without local model files.
    if PREVIEW_MODE:
        print(f"[Preview Mode] Skipping DCGAN checkpoint load: {path}")
        continue
    if not os.path.exists(path):
        print(f"[Missing] DCGAN checkpoint not found for {label}: {path}")
        continue
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    dcgan_generators[label] = model

# Load ProGAN models
progan_generators = {}
for label, path in model_paths["ProGAN"].items():
    model = Generator_ProGAN(256, 256, 3).to(torch.device('cpu'))
    if PREVIEW_MODE:
        print(f"[Preview Mode] Skipping ProGAN checkpoint load: {path}")
        continue
    if not os.path.exists(path):
        print(f"[Missing] ProGAN checkpoint not found for {label}: {path}")
        continue
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    progan_generators[label] = model

# Load StyleGAN2 models
stylegan2_generators = {}
stylegan2_mapping_networks = {}
for label, paths in model_paths["StyleGAN2"].items():
    # Keep StyleGAN2 inference on CPU to match other generators and avoid mixed-device tensors.
    gen_model = Generator_SG2(log_resolution=8, W_DIM=256).to(torch.device('cpu'))
    map_net = MappingNetwork(256, 256).to(torch.device('cpu'))
    if PREVIEW_MODE:
        print(f"[Preview Mode] Skipping StyleGAN2 checkpoint load: {paths}")
        continue
    if not os.path.exists(paths["generator"]) or not os.path.exists(paths["mapping"]):
        print(f"[Missing] StyleGAN2 checkpoints not found for {label}: {paths}")
        continue
    gen_model.load_state_dict(torch.load(paths["generator"], map_location=torch.device('cpu')))
    map_net.load_state_dict(torch.load(paths["mapping"], map_location=torch.device('cpu')))
    gen_model.eval()
    map_net.eval()
    stylegan2_generators[label] = gen_model
    stylegan2_mapping_networks[label] = map_net

# Load WGAN models with weights_only and strict=False
wgan_generators = {}
for label, path in model_paths["WGANs"].items():
    model = Generator_WGAN().to(torch.device('cpu'))
    try:
        if PREVIEW_MODE:
            print(f"[Preview Mode] Skipping WGAN checkpoint load: {path}")
            continue
        if not os.path.exists(path):
            print(f"[Missing] WGAN checkpoint not found for {label}: {path}")
            continue
        # Load the state dict with weights_only=True
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)  # Allows partial compatibility
        model.eval()
        wgan_generators[label] = model
    except FileNotFoundError:
        print(f"Checkpoint file not found for {label}: {path}")
    except RuntimeError as e:
        print(f"Error loading WGAN model for {label}: {e}")


# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about_us")
def about_us():
    return render_template("About_us.html")

@app.route("/generate_info")
def generate_info():
    return render_template("generate.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/detect_info")
def detect_info():
    return render_template("detect.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.form
    model_type = data.get("model")  # "DCGANs", "Progressive GANs", "StyleGAN2", or "WGAN"
    class_name = data.get("class_name")
    num_images = int(data.get("num_images", 1))

    # Select the appropriate model
    if model_type == "DCGANs":
        generators = dcgan_generators
        generation_function = generate_examples_DCGAN
        noise = torch.randn(num_images, Z_DIM, 1, 1).to(torch.device('cpu'))
    elif model_type == "Progressive GANs":
        generators = progan_generators
        generation_function = generate_examples_ProGAN
        noise = torch.randn(num_images, Z_DIM, 1, 1).to(torch.device('cpu'))
    elif model_type == "StyleGAN2":
        generators = stylegan2_generators
        mapping_networks = stylegan2_mapping_networks
        generation_function = generate_examples_SG2
    elif model_type == "WGANs":
        generators = wgan_generators
        generation_function = generate_examples_WGAN
        noise = torch.randn(num_images, 256, 1, 1).to(torch.device('cpu'))
    else:
        return jsonify({"error": "Invalid model type"}), 400

    if not generators:
        return jsonify({
            "error": f"No checkpoints loaded for {model_type}. "
                     f"Disable preview mode or put model files under models/."
        }), 503

    if class_name not in generators:
        return jsonify({
            "error": f"Invalid or unavailable class name for {model_type}: {class_name}",
            "available_classes": sorted(list(generators.keys()))
        }), 400

    if model_type == "StyleGAN2":
        generator = generators[class_name]
        mapping_net = mapping_networks[class_name]
        images_base64, image_buffers = generation_function(generator, mapping_net, num_images)
    else:
        generator = generators[class_name]
        images_base64, image_buffers = generation_function(generator, noise, num_images)

    # Create ZIP file for download
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, buf in enumerate(image_buffers):
            if buf:  # Ensure buffer is not empty
                zip_file.writestr(f"generated_image_{i + 1}.png", buf.getvalue())
    zip_buffer.seek(0)

    # Render template with images and ZIP file link
    return render_template("results.html", images=images_base64, zip_file=True)

@app.route("/download_zip", methods=["GET"])
def download_zip():
    """Route to download the ZIP file containing all generated images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, image_base64 in enumerate(app.config.get("images_base64", [])):
            img_data = base64.b64decode(image_base64)
            zip_file.writestr(f"generated_image_{i + 1}.png", img_data)
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="generated_images.zip"
    )

@app.route("/detect", methods=["POST"])
def detect():
    try:
        # Define paths and device
        model_path = "models/vit-35-Epochs-92-NTP-model.pth"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the uploaded file
        file = request.files.get("file")
        if not file:
            print("No file uploaded.")
            return jsonify({"error": "No file uploaded"}), 400

        # Save the uploaded file temporarily in the static folder
        file_path = os.path.join("static", "temp_image.jpg")
        os.makedirs("static", exist_ok=True)  # Ensure the directory exists
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Initialize the detection app
        detection_app = TumorDetectionApp(model_path=model_path, device=DEVICE)
        print("Detection app initialized.")

        # Predict the class
        predicted_class = detection_app.predict_image(file_path)
        if predicted_class is None:
            print("Prediction failed.")
            return jsonify({"error": "Prediction failed"}), 500

        # Map the prediction to a class name
        class_mapping = {
            0: "Glioma",
            1: "Meningioma",
            2: "No Tumor",
            3: "Pituitary"
        }
        result = class_mapping.get(predicted_class, "Unknown")
        print(f"Prediction successful. Result: {result}")

        # Serve results with the relative path
        return render_template("results-detect.html", images=["temp_image.jpg"], result=result)

    except Exception as e:
        print(f"Error in /detect route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
