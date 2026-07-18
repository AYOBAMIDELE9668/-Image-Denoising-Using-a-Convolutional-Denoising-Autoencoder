# -Image-Denoising-Using-a-Convolutional-Denoising-Autoencoder
A deep learning project that implements a Convolutional Denoising Autoencoder (DAE) in PyTorch to remove Gaussian noise from handwritten digit images. The model learns robust feature representations by reconstructing clean images from noisy inputs.
# Overview

Image denoising is one of the fundamental tasks in computer vision and representation learning. Instead of learning to reconstruct an image directly, a denoising autoencoder learns to recover the original image after it has been intentionally corrupted with noise.

This project demonstrates how a convolutional neural network can learn meaningful latent representations that preserve important image features while filtering out unwanted noise.

The concepts implemented in this project form the foundation of modern generative AI models such as Diffusion Models, Denoising Diffusion Probabilistic Models (DDPM), Stable Diffusion, and image restoration systems.

---

# Project Objectives

The objectives of this project were to:

- Build a Convolutional Denoising Autoencoder from scratch using PyTorch.
- Corrupt clean images by adding Gaussian noise.
- Train the model to reconstruct the original clean images.
- Learn robust latent representations that are resistant to noise.
- Evaluate the reconstruction performance using Binary Cross Entropy (BCE) Loss.
- Understand the role of denoising in modern generative AI.

---

# Dataset

## MNIST Handwritten Digit Dataset

The project uses the MNIST handwritten digit dataset provided by TorchVision.

### Dataset Information

| Property | Value |
|----------|------:|
| Training Images | 60,000 |
| Test Images | 10,000 |
| Classes | 10 |
| Image Size | 28 × 28 |
| Channels | 1 (Grayscale) |
| Source | TorchVision |

The dataset was downloaded automatically during training.

---

# Data Preprocessing

The following preprocessing steps were applied:

- Converted images to tensors.
- Normalized pixel values into the range `[0,1]`.
- Added Gaussian noise to every training image.
- Clamped noisy pixel values between 0 and 1.

Noise generation formula:

```text
Noisy Image = Original Image + Gaussian Noise
```

---

# Model Architecture

The model consists of two major components.

## Encoder

The encoder compresses noisy images into a compact latent representation.

```text
Input Image

↓

Conv2D (1 → 32)

↓

Batch Normalization

↓

ReLU

↓

Max Pooling

↓

Conv2D (32 → 64)

↓

Batch Normalization

↓

ReLU

↓

Max Pooling

↓

Latent Representation
```

---

## Decoder

The decoder reconstructs the clean image from the latent representation.

```text
Latent Representation

↓

ConvTranspose2D (64 → 32)

↓

Batch Normalization

↓

ReLU

↓

ConvTranspose2D (32 → 16)

↓

Batch Normalization

↓

ReLU

↓

Conv2D (16 → 1)

↓

Sigmoid

↓

Reconstructed Image
```

---

# Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- tqdm

---

# Training Configuration

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 128 |
| Epochs | 20 |
| Loss Function | Binary Cross Entropy (BCE) |
| Noise Type | Gaussian Noise |
| Device | CPU / GPU (CUDA Supported) |

---

# Training Results

The model demonstrated stable convergence throughout training.

| Epoch | Training Loss |
|------:|--------------:|
| 1 | 0.134132 |
| 2 | 0.075090 |
| 3 | 0.073152 |
| 4 | 0.072286 |
| 5 | 0.071813 |
| 6 | 0.071484 |
| 7 | 0.071253 |
| 8 | 0.071053 |
| 9 | 0.070936 |
| 10 | 0.070781 |
| 11 | 0.070688 |
| 12 | 0.070553 |
| 13 | 0.070517 |
| 14 | 0.070444 |
| 15 | 0.070420 |
| 16 | 0.070306 |
| 17 | 0.070265 |
| 18 | 0.070226 |
| 19 | 0.070174 |
| 20 | 0.070173 |

---

# Performance Analysis

The model successfully reduced the training loss from:

```text
0.134132
```

to

```text
0.070173
```

This represents an approximate **47.7% reduction** in reconstruction loss during training.

The steady decrease in loss indicates that the model successfully learned to map noisy handwritten digit images back to their clean counterparts.

---

# Model Output

The trained model produces three outputs for visual comparison:

```text
Original Image

↓

Noisy Image

↓

Recovered Image
```

The recovered image demonstrates the model's ability to remove Gaussian noise while preserving the structural characteristics of handwritten digits.

---

# Features Implemented

- Automatic MNIST dataset download
- Gaussian noise generation
- Convolutional encoder
- Convolutional decoder
- Batch Normalization
- Binary Cross Entropy loss
- Adam optimizer
- Learning rate scheduler
- GPU support
- Automatic model checkpoint saving
- Best model selection
- Image visualization
- Inference on unseen test images

---

# What I Learned

This project provided practical experience in:

- Building denoising autoencoders.
- Learning robust image representations.
- Working with convolutional encoder-decoder architectures.
- Applying Gaussian noise as a data corruption technique.
- Training deep neural networks for image restoration.
- Using transpose convolutions for image reconstruction.
- Implementing Batch Normalization.
- Applying learning rate scheduling.
- Saving and loading PyTorch model checkpoints.
- Performing image reconstruction on unseen data.

---

# Applications

Denoising Autoencoders are widely used in:

- Medical image enhancement
- MRI reconstruction
- Image restoration
- Image compression
- Representation learning
- Feature extraction
- Anomaly detection
- Noise reduction in satellite imagery
- Document restoration
- Face enhancement
- Low-light image enhancement
- Pretraining deep neural networks

---

# Relationship to Modern Generative AI

Denoising Autoencoders are closely related to diffusion-based generative models.

In this project, the model learns to remove Gaussian noise from an image in a single forward pass.

Modern diffusion models follow the same principle but perform denoising through a sequence of many small iterative steps, enabling the generation of entirely new images from random noise.

This project establishes the conceptual foundation for:

- Variational Autoencoders (VAEs)
- Diffusion Models (DDPM)
- Latent Diffusion Models
- Stable Diffusion
- Image-to-Image Translation
- Image Inpainting

---

# Future Improvements

Potential enhancements include:

- Residual blocks
- Skip connections (U-Net architecture)
- Structural Similarity Index (SSIM) evaluation
- Peak Signal-to-Noise Ratio (PSNR) evaluation
- Mixed precision (AMP) training
- TensorBoard integration
- Early stopping
- Validation dataset evaluation
- Colored image denoising
- Training on CelebA and CIFAR-10 datasets
- Integration with diffusion-based architectures

---

# Repository Structure

```text
image-denoising-autoencoder/
│
├── data/
├── denoising_autoencoder.py
├── best_denoising_autoencoder.pth
├── README.md
├── requirements.txt
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/image-denoising-autoencoder.git
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

# Usage

Run the training script.

```bash
python denoising_autoencoder.py
```

The script automatically:

- Downloads the MNIST dataset.
- Generates noisy training images.
- Trains the denoising autoencoder.
- Saves the best-performing model.
- Displays original, noisy, and reconstructed images.

---

# Project Outcome

This project successfully developed a Convolutional Denoising Autoencoder capable of learning robust latent representations for image restoration.

After training for **20 epochs**, the model reduced the Binary Cross Entropy loss from **0.134132** to **0.070173**, demonstrating stable convergence and effective denoising performance on the MNIST dataset.

The implementation establishes a strong foundation for future work in probabilistic latent variable models, diffusion-based image generation, and state-of-the-art generative AI architectures.
