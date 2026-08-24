'''
By ChatGPT:
Here’s a Python function to add Gaussian noise to an MNIST image while
maintaining a target Peak Signal-to-Noise Ratio (PSNR) in dB.
'''
import numpy as np
import cv2

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def add_gaussian_noise_psnr(image, target_psnr_db):
    """
    Adds Gaussian noise to an MNIST image to achieve the specified PSNR in dB.

    Parameters:
    - image: numpy array (HxW), grayscale MNIST image with pixel values in range [0, 255].
    - target_psnr_db: float, desired PSNR value in dB.

    Returns:
    - noisy_image: numpy array (HxW), MNIST image with added Gaussian noise.
    """
    # Ensure image is float for processing
    image = image.astype(np.float32)

    # Compute the power of the original image
    mse_target = (255 ** 2) / (10 ** (target_psnr_db / 10))

    # Standard deviation of the noise
    sigma = np.sqrt(mse_target)

    # Generate Gaussian noise
    noise = np.random.normal(0, sigma, image.shape)

    # Add noise to image
    noisy_image = image + noise

    # Clip values to maintain valid image range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# Load sample MNIST image
(x_train, _), _ = mnist.load_data()
mnist_image = x_train[0]  # First image in the dataset

# Add Gaussian noise with target PSNR = 20 dB
noisy_mnist = add_gaussian_noise_psnr(mnist_image, 10)

# Display images
plt.subplot(1, 2, 1)
plt.title("Original MNIST")
plt.imshow(mnist_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Noisy MNIST (20dB)")
plt.imshow(noisy_mnist, cmap="gray")
plt.axis("off")

plt.show()
