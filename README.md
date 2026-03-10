[![CI](https://github.com/mozaloom/medgan/actions/workflows/main.yml/badge.svg)](https://github.com/mozaloom/medgan/actions/workflows/main.yml)
[![Docker Image CI](https://github.com/mozaloom/medgan/actions/workflows/push-docker.yml/badge.svg)](https://github.com/mozaloom/medgan/actions/workflows/push-docker.yml)

# MedGAN: Advanced Medical Image Generation

<img src="static/css/Blue_ABstract_Brain_Technology_Logo__1_-removebg-preview.png" alt="medgan Logo" width="120" style="margin-bottom: 20px;">

## Overview

MedGAN is a comprehensive framework for generating high-quality synthetic medical images using state-of-the-art Generative Adversarial Networks (GANs). The project focuses on brain tumor MRI scans and includes implementations of multiple cutting-edge GAN architectures optimized for medical imaging applications.

## Features

- **Multiple GAN Implementations:**
  - DCGAN (Deep Convolutional GAN)
  - ProGAN (Progressive Growing of GANs)
  - StyleGAN2 (Style-based Generator with improvements)
  - WGAN (Wasserstein GAN with gradient penalty)

- **Web Application Interface:**
  - Generate synthetic brain MRI scans
  - Detect tumor types from uploaded MRI images
  - Interactive and user-friendly interface

- **Pre-trained Models:**
  - Models for three tumor types: Glioma, Meningioma, and Pituitary
  - ViT-based tumor detection model (92% accuracy)

## Architecture Performance Comparison

| Architecture | Image Quality | Training Stability | Generation Diversity | Training Speed |
|--------------|---------------|--------------------|-----------------------|---------------|
| ProGAN       | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐            | ⭐⭐⭐        |
| StyleGAN2    | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐          | ⭐⭐         |
| WGAN-GP      | ⭐⭐⭐       | ⭐⭐⭐⭐          | ⭐⭐⭐              | ⭐⭐⭐⭐     |
| DCGAN        | ⭐⭐⭐       | ⭐⭐⭐            | ⭐⭐                | ⭐⭐⭐⭐⭐   |

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 1.9+
- Flask (for web application)
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mozaloom/medgan.git
cd medgan
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the web application:
```bash
python app.py
```

4. Access the web interface at `http://localhost:5000`

## Usage

### Web Application

The MedGAN web application offers two primary functionalities:

1. **Generate synthetic brain MRI scans:**
   - Select tumor type (Glioma, Meningioma, Pituitary)
   - Choose GAN architecture
   - Generate high-quality synthetic MRI images

2. **Detect tumor types:**
   - Upload brain MRI scans
   - Receive AI-powered tumor classification
   - View detection confidence scores


Check the individual model implementation files for specific training parameters.

## Project Structure

```
medgan/
├── app.py                   # Flask web application
├── medgan/                  # Core GAN implementations
│   ├── dcgan.py
│   ├── progan.py
│   ├── stylegan.py
│   ├── wgan.py
│   └── vit.py
├── models/                  # Pre-trained model weights
├── notebooks/               # Training notebooks
│   ├── dcgan/
│   ├── progan/
│   ├── stylegan/
│   └── wgan/
├── static/                  # Web assets
└── templates/               # HTML templates
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) from Kaggle
- Research papers implementing the original GAN architectures:
  - [DCGAN](https://arxiv.org/abs/1511.06434)
  - [ProGAN](https://arxiv.org/abs/1710.10196)
  - [StyleGAN2](https://arxiv.org/abs/1912.04958)
  - [WGAN](https://arxiv.org/abs/1701.07875)