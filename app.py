
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# --- Model Loading and Setup ---
# The GAN generator expects a latent space input of size 32 (based on your code)
LATENT_DIMENSION = 32 

# Attempt to load the saved generator model.
# If the file is missing, we use a mock generator so the UI still loads.
try:
    # IMPORTANT: Ensure 'sign_language_gan_generator.keras' is uploaded to your Colab session
    generator = tf.keras.models.load_model("sign_language_gan_generator.keras")
    MODEL_LOADED = True
    # The output shape of the model's images (assuming 64x64 or similar grayscale output)
    OUTPUT_SHAPE = generator.output_shape[1:3] 
    print("Generator model loaded successfully from disk.")
except Exception as e:
    print(f"WARNING: Could not load the model 'sign_language_gan_generator.keras'. Error: {e}")
    print("Using a mock generator. Generated images will be blank white/gray squares.")
    MODEL_LOADED = False
    OUTPUT_SHAPE = (64, 64) # Assume a default output size for the mock

    # Define a mock generator function that returns noise-free data if the model fails to load.
    def mock_generator(noise, training=False):
        # Return a batch of slightly off-white noise-like images (64x64x1)
        batch_size = tf.shape(noise)[0]
        # Return values in the range [-1, 1] to simulate the generator's tanh output
        return tf.ones((batch_size, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], 1), dtype=tf.float32) * 0.5 
    generator = mock_generator


def generate_hands(num_images, seed=None):
    """
    Generates hand images using the GAN model and converts them to PIL images for Gradio.
    """
    # Cast inputs to appropriate types
    num_images = int(num_images) if num_images else 16
    
    # Set seed if provided
    if seed is not None:
        # TensorFlow seeds are global, so this applies to the random normal generation
        tf.random.set_seed(int(seed))
        
    # 1. Generate noise vector from latent space
    noise = tf.random.normal([num_images, LATENT_DIMENSION])

    # 2. Generate images
    # We use a try-except block just in case the mock generator definition has subtle differences
    try:
        generated_images = generator(noise, training=False)
    except Exception:
        generated_images = generator(noise)
        
    # 3. Denormalize from tanh output [-1, 1] to pixel values [0, 255]
    generated_images = (generated_images + 1) * 127.5
    
    # 4. Clip and convert to unsigned 8-bit integers
    generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)
    
    # 5. Convert to a list of PIL images for the Gradio Gallery
    pil_images = []
    for i in range(num_images):
        # The output is (Batch, H, W, 1). We slice out the channel dimension (index 0)
        img_array = generated_images[i, :, :, 0]
        # 'L' mode is for 8-bit grayscale images
        pil_img = Image.fromarray(img_array, mode="L")
        pil_images.append(pil_img)
    
    return pil_images


# --- Gradio Interface Definition ---
with gr.Blocks(title="Sign Language Hand Generator (GAN)") as demo:
    gr.Markdown(f"""
    # 👐 GAN-based Sign Language Hand Image Generator
    
    This application uses a trained Generative Adversarial Network (GAN) 
    to synthesize realistic grayscale hand images.
    
    **Model Status:** {'✅ Loaded' if MODEL_LOADED else '❌ Mock Generator Active (Upload your model!)'}
    """)
    
    with gr.Row():
        # Input for number of images (n in your function)
        num_images = gr.Number(
            label="Number of Images (Max 64)", 
            value=16, 
            minimum=1, 
            maximum=64,
            precision=0
        )
        
        # Input for the random seed
        seed = gr.Number(
            label="Random Seed (Optional)", 
            value=None, 
            minimum=0, 
            precision=0
        )
    
    generate_btn = gr.Button("Generate Hand Images", variant="primary")
    
    # Output Gallery
    output_gallery = gr.Gallery(
        label=f"Generated Hands ({OUTPUT_SHAPE[0]}x{OUTPUT_SHAPE[1]} grayscale)", 
        columns=4, 
        height="auto",
        object_fit="contain"
    )
    
    # Define the click behavior
    generate_btn.click(
        fn=generate_hands,
        inputs=[num_images, seed],
        outputs=output_gallery
    )

# --- Launch the App ---
if __name__ == "__main__":
    # Launch with share=True for a public URL in Colab
    demo.launch(share=True)