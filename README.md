# Sign Language Hand Generator GAN ✋

**Generate unlimited realistic sign language hand poses using a Deep Convolutional GAN (DCGAN)**  

> A from-scratch trained **DCGAN** that generates high-quality 28×28 grayscale hand images performing various sign language gestures — trained on real sign language dataset.

---

### Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic images of human hands in sign language positions.

- Input: Random noise vector (latent dimension = 32)
- Output: 28×28 grayscale hand image (upscaled to 128×128 for crisp display)
- Architecture: Fully convolutional Generator + Discriminator with LeakyReLU, BatchNorm, and Dropout
- Framework: TensorFlow 2 + Keras
- Interactive Demo: Built with **Gradio** (one-click generation in browser)

Perfect for **accessibility tools**, **data augmentation**, **education**, or just showing off cool generative AI!

---

### Features

- Clean DCGAN implementation from scratch
- High-quality hand generation after ~60 epochs
- Real-time web demo with Gradio (`app.py`)
- 128×128 upscaling using Lanczos filter for sharp results
- Reproducible results with seed control
- Full training notebook included (`C4W4_Assignment.ipynb`)

---

### Generated Examples (Seed = 42)

<div align="center">
  <img src="https://github.com/Nauman123-coder/Sign-Language-Hand-Generator-GAN/blob/main/example_grid.png?raw=true" width="600"/>
</div>

---

### How It Was Built

1. **Dataset**: Real sign language hand images (28×28 grayscale)
2. **Model**:
   - Generator: Dense → Reshape → Conv2DTranspose (SELU + BatchNorm) → Tanh
   - Discriminator: Conv2D (LeakyReLU) → Dropout → Dense → Sigmoid
3. **Training**: Alternating discriminator & generator updates using `RMSprop`
4. **Trained for 60+ epochs** until hands became sharp and recognizable
5. **Saved best generator** → deployed via Gradio

---

### Performance

| Metric                  | Result                          |
|-------------------------|----------------------------------|
| Image Resolution        | 28×28 → upscaled to 128×128     |
| Latent Dimension        | 32                              |
| Training Epochs         | ~60                             |
| Visual Quality          | High (clear fingers & poses)    |
| Inference Speed         | < 0.1s for 16 images (on GPU)   |

---

### Run the Live Demo

```bash
python app.py
